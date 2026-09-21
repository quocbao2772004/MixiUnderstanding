#!/usr/bin/env python3
"""Distill clean-component semantics into a mixture-span adapter on V5.

The student sees only frozen mixture-span features.  Clean source statistics
are used only as training targets and are never required at inference.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import LongHeadConfig, LongSemanticHead
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
DATA = Path("/var/tmp/qces_full188_tiered_realistic_v5")
NUM_CLASSES = 188


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_main_train.pt")
    parser.add_argument("--dev-cache", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_main_dev.pt")
    parser.add_argument("--clean-train-cache", type=Path, default=BASE / "long_short_semantic_teacher_v2/full_source_stats_train.pt")
    parser.add_argument("--clean-dev-cache", type=Path, default=BASE / "long_short_semantic_teacher_v2/full_source_stats_dev.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=BASE / "long_short_semantic_teacher_v2/long_short_semantic_teacher_v2_best.pt")
    parser.add_argument("--train-manifest", type=Path, default=DATA / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=DATA / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASE / "v5_paired_clean_distill_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6801)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.30)
    parser.add_argument("--embedding-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class PairedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, payload: Mapping[str, torch.Tensor]) -> None: self.payload = payload
    def __len__(self) -> int: return int(self.payload["mixture_input"].shape[0])
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.payload.items()}


class CleanDenoisingAdapter(nn.Module):
    """Residual adapter initialized as the frozen mixture teacher."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float, frozen_residual: nn.Linear) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.frozen_residual = frozen_residual
        self.frozen_residual.requires_grad_(False)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.embedding_delta = nn.Linear(hidden_dim, embedding_dim)
        self.logit_delta = nn.Linear(hidden_dim, NUM_CLASSES)
        nn.init.zeros_(self.embedding_delta.weight); nn.init.zeros_(self.embedding_delta.bias)
        nn.init.zeros_(self.logit_delta.weight); nn.init.zeros_(self.logit_delta.bias)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # V3 cache layout: stats(2304), long embedding(256), long logits(188), R1 logits(188).
        mix_embedding = value[:, 2304 : 2304 + self.embedding_dim].float()
        mix_r1 = value[:, -NUM_CLASSES:].float()
        hidden = self.trunk(value.float())
        embedding = F.normalize(mix_embedding + self.embedding_delta(hidden), dim=-1)
        logits = mix_r1 + self.frozen_residual(embedding) + self.logit_delta(hidden)
        return logits, embedding


def source_ids_for_cache(cache: Mapping[str, Any], manifest: Path) -> list[str]:
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); rows[str(row["scene_id"])] = row
    result = []
    for scene_id in cache["scene_id"]:
        result.extend(
            str(event["source_id"]) for event in rows[str(scene_id)]["events"]
            if str(event.get("event_kind", "semantic")) == "semantic"
        )
    return result


@torch.inference_mode()
def build_payload(
    mixture_cache: Mapping[str, Any], clean_cache: Mapping[str, Any], manifest: Path,
    teacher: LongSemanticHead, device: torch.device,
) -> dict[str, torch.Tensor]:
    exact = mixture_cache["oracle_exact"]
    mixture_input = mixture_cache["oracle_input"][exact]
    labels = mixture_cache["oracle_labels"][exact].long()
    source_ids = source_ids_for_cache(mixture_cache, manifest)
    if len(source_ids) != len(labels): raise ValueError("source/event mismatch")
    position = {str(value): index for index, value in enumerate(clean_cache["source_id"])}
    paired = torch.tensor([value in position for value in source_ids], dtype=torch.bool)
    clean_logits = torch.zeros(len(labels), NUM_CLASSES, dtype=torch.float16)
    embedding_dim = int(teacher.config.embedding_dim)
    clean_embeddings = torch.zeros(len(labels), embedding_dim, dtype=torch.float16)
    paired_rows = torch.where(paired)[0]
    source_rows = torch.tensor([position[source_ids[index]] for index in paired_rows.tolist()], dtype=torch.long)
    teacher.eval()
    for start in range(0, len(paired_rows), 512):
        destination = paired_rows[start : start + 512]
        selected = source_rows[start : start + 512]
        logits, embeddings = teacher(
            clean_cache["fixed_stats"][selected].to(device), clean_cache["r1_logits"][selected].to(device)
        )
        clean_logits[destination] = logits.cpu().half(); clean_embeddings[destination] = embeddings.cpu().half()
    return {
        "mixture_input": mixture_input,
        "label": labels,
        "paired": paired,
        "clean_logits": clean_logits,
        "clean_embedding": clean_embeddings,
    }


@torch.inference_mode()
def evaluate(model: CleanDenoisingAdapter, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval(); logits_all = []; labels_all = []; paired_all = []
    for batch in loader:
        logits, _ = model(batch["mixture_input"].to(device)); logits_all.append(logits.cpu())
        labels_all.append(batch["label"]); paired_all.append(batch["paired"])
    logits = torch.cat(logits_all); labels = torch.cat(labels_all).long(); paired = torch.cat(paired_all).bool()
    def subset(mask: torch.Tensor) -> dict[str, Any]:
        current = logits[mask]; target = labels[mask]; order = current.argsort(-1, descending=True)
        totals = Counter(target.tolist()); correct = Counter()
        for gold, pred in zip(target.tolist(), order[:, 0].tolist()): correct[gold] += int(gold == pred)
        return {
            "events": int(mask.sum()),
            "top1_accuracy_\u2191": float(order[:, 0].eq(target).float().mean()),
            "top5_accuracy_\u2191": float((order[:, :5] == target[:, None]).any(-1).float().mean()),
            "macro_top1_accuracy_\u2191": float(sum(correct[key] / value for key, value in totals.items()) / len(totals)),
        }
    return {"all": subset(torch.ones_like(paired)), "paired": subset(paired), "unpaired": subset(~paired)}


def baseline_metrics(payload: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    value = payload["mixture_input"].float(); labels = payload["label"].long(); paired = payload["paired"].bool()
    logits = value[:, 2304 + 256 : 2304 + 256 + NUM_CLASSES]
    def score(mask: torch.Tensor) -> dict[str, Any]:
        order = logits[mask].argsort(-1, descending=True); target = labels[mask]
        return {"events": int(mask.sum()), "top1_accuracy_\u2191": float(order[:, 0].eq(target).float().mean()), "top5_accuracy_\u2191": float((order[:, :5] == target[:, None]).any(-1).float().mean())}
    return {"all": score(torch.ones_like(paired)), "paired": score(paired), "unpaired": score(~paired)}


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
    teacher = LongSemanticHead(LongHeadConfig(**teacher_payload["long_config"]))
    teacher.load_state_dict(teacher_payload["long_model_state_dict"], strict=True); teacher.to(device).eval()
    payloads = {}
    for split, mix_path, clean_path, manifest in (
        ("train", args.train_cache, args.clean_train_cache, args.train_manifest),
        ("dev", args.dev_cache, args.clean_dev_cache, args.dev_manifest),
    ):
        payloads[split] = build_payload(
            torch.load(mix_path, map_location="cpu", weights_only=True),
            torch.load(clean_path, map_location="cpu", weights_only=True), manifest, teacher, device,
        )
        print(json.dumps({"payload": split, "events": len(payloads[split]["label"]), "paired": int(payloads[split]["paired"].sum())}), flush=True)
    frozen_residual = nn.Linear(teacher.config.embedding_dim, NUM_CLASSES)
    frozen_residual.load_state_dict(teacher.residual_head.state_dict(), strict=True)
    del teacher
    model = CleanDenoisingAdapter(
        int(payloads["train"]["mixture_input"].shape[-1]), args.hidden_dim,
        int(teacher_payload["long_config"]["embedding_dim"]), args.dropout, frozen_residual,
    ).to(device)
    train_loader = DataLoader(PairedDataset(payloads["train"]), batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    dev_loader = DataLoader(PairedDataset(payloads["dev"]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    labels = payloads["train"]["label"].long(); counts = torch.bincount(labels, minlength=NUM_CLASSES).float().clamp_min(1)
    class_weight = ((counts.median() / counts).sqrt().clamp(0.6, 2.0)).to(device)
    optimizer = torch.optim.AdamW((value for value in model.parameters() if value.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03)
    history = []; best = None; best_epoch = 0; best_state = None; stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = total_ce = total_kl = total_emb = seen = paired_seen = 0
        for batch in train_loader:
            value = batch["mixture_input"].to(device); target = batch["label"].long().to(device); paired = batch["paired"].bool().to(device)
            logits, embedding = model(value)
            ce = F.cross_entropy(logits, target, weight=class_weight, label_smoothing=0.02)
            if paired.any():
                teacher_logits = batch["clean_logits"].to(device)[paired].float(); teacher_embedding = batch["clean_embedding"].to(device)[paired].float()
                temperature = args.temperature
                kl = F.kl_div((logits[paired] / temperature).log_softmax(-1), (teacher_logits / temperature).softmax(-1), reduction="batchmean") * temperature**2
                emb = (1.0 - F.cosine_similarity(embedding[paired], teacher_embedding, dim=-1)).mean()
            else:
                kl = logits.sum() * 0.0; emb = logits.sum() * 0.0
            loss = args.ce_weight * ce + args.kl_weight * kl + args.embedding_weight * emb
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            size = target.numel(); total_loss += float(loss.detach()) * size; total_ce += float(ce.detach()) * size
            total_kl += float(kl.detach()) * int(paired.sum()); total_emb += float(emb.detach()) * int(paired.sum()); seen += size; paired_seen += int(paired.sum())
        scheduler.step(); current = evaluate(model, dev_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total_loss/seen, "train_ce": total_ce/seen, "train_kl_paired": total_kl/max(paired_seen,1), "train_embedding_paired": total_emb/max(paired_seen,1), "dev": current}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = (float(current["all"]["top1_accuracy_\u2191"]), float(current["paired"]["top1_accuracy_\u2191"]), float(current["all"]["top5_accuracy_\u2191"]))
        if best is None or key > best:
            best = key; best_epoch = epoch; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True); break
    if best_state is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(best_state); best_metrics = evaluate(model, dev_loader, device)
    checkpoint_path = output_dir / "paired_clean_distill_v1_best.pt"
    _atomic_torch({"format": "qces_v5_paired_clean_distill_checkpoint_v1", "model_state_dict": best_state, "input_dim": int(payloads["train"]["mixture_input"].shape[-1]), "hidden_dim": args.hidden_dim, "embedding_dim": int(teacher_payload["long_config"]["embedding_dim"]), "dropout": args.dropout, "best_epoch": best_epoch, "best_metrics": best_metrics}, checkpoint_path)
    baseline = baseline_metrics(payloads["dev"])
    gates = {
        "all_top1_ge_0_65": float(best_metrics["all"]["top1_accuracy_\u2191"]) >= 0.65,
        "all_top5_ge_0_90": float(best_metrics["all"]["top5_accuracy_\u2191"]) >= 0.90,
        "paired_top1_improves_baseline_by_0_05": float(best_metrics["paired"]["top1_accuracy_\u2191"]) >= float(baseline["paired"]["top1_accuracy_\u2191"]) + 0.05,
    }
    receipt = {
        "format": "qces_v5_paired_clean_distill_training_receipt_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen mixture features to clean-teacher embedding/logit distillation; clean unavailable at inference",
        "qa_question_or_answer_used_as_input": False, "source_identity_overlap_train_dev": 0,
        "data": {split: {"events": len(value["label"]), "paired_events": int(value["paired"].sum()), "coverage": float(value["paired"].float().mean())} for split, value in payloads.items()},
        "baseline_mixture_teacher": baseline, "best_epoch": best_epoch, "best_metrics": best_metrics, "history": history,
        "gates": gates, "decision": "integrate_distilled_semantics_with_predicted_slots" if all(gates.values()) else "paired_distillation_insufficient",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256_file(checkpoint_path),
        "teacher_checkpoint_sha256": _sha256_file(args.teacher_checkpoint.resolve()),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "baseline": baseline, "best_metrics": best_metrics, "gates": gates, "decision": receipt["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
