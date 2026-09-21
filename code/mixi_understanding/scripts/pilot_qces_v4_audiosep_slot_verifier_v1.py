#!/usr/bin/env python3
"""Pilot frozen-AudioSep verification of V4 semantic slot shortlists.

The class-agnostic localizer first proposes intervals and the frozen semantic
head supplies Top-K labels.  AudioSep is queried only for that shortlist.  Each
returned stem is independently rescored by the frozen BEATs-R1 detector inside
the proposed interval.  The fixed deployed score is an equal-weight sum of
within-slot z-normalized proposal and verifier scores.

The cohort is selected by SHA-256(scene_id), before inference and without QA or
gold-label access.  Per-scene fragments make the expensive pilot resumable.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy.optimize import linear_sum_assignment

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    describe,
    encode_prompts,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_model,
    strong_logits_from_waveform,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL,
    ResidualSlotSemanticHead,
)


FORMAT = "qces_v4_audiosep_slot_verifier_pilot_v1"
FRAGMENT_FORMAT = "qces_v4_audiosep_slot_verifier_fragment_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    semantic = base / "v4_slot_semantic_head_v1"
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot-cache", type=Path, default=semantic / "frozen_slots_dev.pt")
    parser.add_argument("--semantic-checkpoint", type=Path, default=semantic / "v4_slot_semantic_head_v1_best.pt")
    parser.add_argument("--scene-manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--pretrainedsed-checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep")
    parser.add_argument("--audiosep-config", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml")
    parser.add_argument("--audiosep-checkpoint", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin")
    parser.add_argument("--output-dir", type=Path, default=base / "v4_audiosep_slot_verifier_pilot_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--separator-batch-size", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--prompt-template", default="the sound of {label}")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scene_id = str(row["scene_id"])
            if scene_id in rows:
                raise ValueError(f"duplicate scene_id: {scene_id}")
            rows[scene_id] = row
    return rows


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != NONE_LABEL or len(set(labels)) != len(labels):
        raise ValueError(f"expected {NONE_LABEL} unique labels, got {len(labels)}")
    return labels


def fixed_z(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    return (value - value.mean()) / value.std(unbiased=False).clamp_min(1e-5)


@torch.inference_mode()
def load_semantic_predictions(cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> torch.Tensor:
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]),
        NONE_LABEL, float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    values = cache["slot_input"]
    return torch.cat([model(values[i : i + 256]).cpu() for i in range(0, len(values), 256)])


def selected_slots(
    cache: Mapping[str, Any], logits: torch.Tensor, index: int,
    *, objectness_threshold: float, top_k: int,
) -> list[dict[str, Any]]:
    keep = cache["objectness"][index].float() >= objectness_threshold
    original_indices = torch.where(keep)[0]
    scores = logits[index][keep].float()
    semantic_keep = scores.argmax(dim=-1) != NONE_LABEL
    original_indices = original_indices[semantic_keep]
    scores = scores[semantic_keep, :NONE_LABEL]
    intervals = cache["intervals"][index].float()[keep][semantic_keep]
    result = []
    for slot_index, interval, row in zip(original_indices.tolist(), intervals, scores):
        values, labels = row.topk(top_k)
        result.append({
            "slot_index": int(slot_index),
            "interval": [float(interval[0]), float(interval[1])],
            "candidate_label_ids": labels.tolist(),
            "base_scores": values.tolist(),
        })
    return result


@torch.inference_mode()
def score_scene(
    *,
    scene_id: str,
    row: Mapping[str, Any],
    slots: list[dict[str, Any]],
    labels: Sequence[str],
    prompt_embeddings: Mapping[str, torch.Tensor],
    prompts: Mapping[int, str],
    separator: torch.nn.Module,
    detector: torch.nn.Module,
    device: torch.device,
    separator_batch_size: int,
) -> dict[str, Any]:
    waveform, sample_rate = sf.read(Path(str(row["mixture_path"])), dtype="float32", always_2d=False)
    mixture16 = torch.as_tensor(waveform).float()
    if mixture16.ndim > 1:
        mixture16 = mixture16.mean(dim=-1)
    if int(sample_rate) != 16_000:
        mixture16 = AF.resample(mixture16, int(sample_rate), 16_000)
    target_samples = 160_000
    mixture16 = F.pad(mixture16[:target_samples], (0, max(0, target_samples - mixture16.numel())))
    mixture32 = AF.resample(mixture16, 16_000, 32_000).to(device)

    unique_ids = sorted({label_id for slot in slots for label_id in slot["candidate_label_ids"]})
    verifier_by_label: dict[int, torch.Tensor] = {}
    energy_by_label: dict[int, torch.Tensor] = {}
    for start in range(0, len(unique_ids), separator_batch_size):
        batch_ids = unique_ids[start : start + separator_batch_size]
        condition = torch.stack([prompt_embeddings[prompts[label_id]] for label_id in batch_ids]).to(device)
        mixtures = mixture32[None, None].expand(len(batch_ids), 1, -1)
        stems32 = separator({"mixture": mixtures, "condition": condition})["waveform"][:, 0]
        stems16 = AF.resample(stems32, 32_000, 16_000)
        detector_logits = strong_logits_from_waveform(detector, stems16).float()
        for batch_index, label_id in enumerate(batch_ids):
            verifier_by_label[label_id] = detector_logits[batch_index, :, label_id].detach().cpu()
            energy_by_label[label_id] = stems16[batch_index].square().detach().cpu()
        del condition, mixtures, stems32, stems16, detector_logits

    output_slots: list[dict[str, Any]] = []
    for slot in slots:
        start_frame = max(0, min(249, int(math.floor(slot["interval"][0] * 250))))
        end_frame = max(start_frame + 1, min(250, int(math.ceil(slot["interval"][1] * 250))))
        start_sample = max(0, min(159_999, int(math.floor(slot["interval"][0] * 160_000))))
        end_sample = max(start_sample + 1, min(160_000, int(math.ceil(slot["interval"][1] * 160_000))))
        verifier_scores = []
        concentrations = []
        for label_id in slot["candidate_label_ids"]:
            verifier_scores.append(float(verifier_by_label[label_id][start_frame:end_frame].mean()))
            energy = energy_by_label[label_id]
            inside = energy[start_sample:end_sample].mean()
            total = energy.mean().clamp_min(1e-10)
            concentrations.append(float(torch.log((inside + 1e-10) / total)))
        base = torch.tensor(slot["base_scores"])
        verifier = torch.tensor(verifier_scores)
        # This fusion rule is fixed before viewing gold labels.
        fusion = fixed_z(base) + fixed_z(verifier)
        result = dict(slot)
        result.update({
            "verifier_scores": verifier_scores,
            "temporal_log_energy_concentration": concentrations,
            "baseline_top1_label_id": int(slot["candidate_label_ids"][0]),
            "verifier_top1_label_id": int(slot["candidate_label_ids"][int(verifier.argmax())]),
            "fixed_fusion_top1_label_id": int(slot["candidate_label_ids"][int(fusion.argmax())]),
        })
        output_slots.append(result)
    return {
        "format": FRAGMENT_FORMAT,
        "scene_id": scene_id,
        "audio_sha256": row.get("audio_sha256"),
        "slots": output_slots,
        "separator_calls": len(unique_ids),
        "unique_candidate_labels": len(unique_ids),
    }


def evaluate(
    cache: Mapping[str, Any], scene_indices: Sequence[int], fragments: Mapping[str, Mapping[str, Any]],
    *, iou_threshold: float,
) -> dict[str, Any]:
    methods = ("baseline_top1_label_id", "verifier_top1_label_id", "fixed_fusion_top1_label_id")
    correct = {name: 0 for name in methods}
    eligible_correct = {name: 0 for name in methods}
    gold_count = localized = eligible = 0
    changes = {"fixed_fusion_corrects_baseline": 0, "fixed_fusion_breaks_baseline": 0, "both_correct": 0, "both_wrong": 0}
    items: list[dict[str, Any]] = []
    for index in scene_indices:
        scene_id = str(cache["scene_id"][index])
        slots = fragments[scene_id]["slots"]
        predicted_intervals = torch.tensor([slot["interval"] for slot in slots], dtype=torch.float32)
        gold_intervals = cache["gold_intervals"][index].float()
        gold_labels = cache["gold_labels"][index].long()
        gold_count += int(gold_intervals.shape[0])
        if predicted_intervals.numel() == 0:
            continue
        iou = interval_iou_matrix(predicted_intervals, gold_intervals)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        for pred_index, gold_index in zip(rows, columns):
            overlap = float(iou[pred_index, gold_index])
            if overlap < iou_threshold:
                continue
            localized += 1
            slot = slots[pred_index]
            gold = int(gold_labels[gold_index])
            candidate_eligible = gold in slot["candidate_label_ids"]
            eligible += int(candidate_eligible)
            flags = {name: int(slot[name]) == gold for name in methods}
            for name in methods:
                correct[name] += int(flags[name])
                eligible_correct[name] += int(candidate_eligible and flags[name])
            if flags[methods[0]] and flags[methods[2]]:
                changes["both_correct"] += 1
            elif flags[methods[0]] and not flags[methods[2]]:
                changes["fixed_fusion_breaks_baseline"] += 1
            elif not flags[methods[0]] and flags[methods[2]]:
                changes["fixed_fusion_corrects_baseline"] += 1
            else:
                changes["both_wrong"] += 1
            items.append({
                "scene_id": scene_id,
                "predicted_interval": slot["interval"],
                "gold_interval": gold_intervals[gold_index].tolist(),
                "iou": overlap,
                "gold_label_id": gold,
                "candidate_eligible": candidate_eligible,
                **{name: int(slot[name]) for name in methods},
            })
    metrics: dict[str, Any] = {
        "scenes": len(scene_indices), "gold_events": gold_count,
        "localized_events_iou50": localized,
        "candidate_eligible_events_top20": eligible,
        "localization_recall_iou50": localized / max(gold_count, 1),
        "candidate_recall_at_20_given_iou50": eligible / max(localized, 1),
        "methods": {}, "paired_changes": changes, "items": items,
    }
    for name in methods:
        metrics["methods"][name] = {
            "top1_given_iou50_\u2191": correct[name] / max(localized, 1),
            "top1_given_iou50_and_candidate_eligible_\u2191": eligible_correct[name] / max(eligible, 1),
            "joint_recall_iou50_\u2191": correct[name] / max(gold_count, 1),
        }
    return metrics


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and args.overwrite:
        # Preserve resumable fragments; overwrite only refreshes the final receipt.
        receipt = output_dir / "receipt.json"
        if receipt.exists():
            receipt.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    fragment_dir = output_dir / "fragments"
    fragment_dir.mkdir(exist_ok=True)

    cache_path = args.slot_cache.resolve()
    semantic_path = args.semantic_checkpoint.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    semantic_checkpoint = torch.load(semantic_path, map_location="cpu", weights_only=True)
    labels = read_labels(args.ontology.resolve())
    if list(semantic_checkpoint["labels"]) != labels:
        raise ValueError("semantic checkpoint and ontology labels differ")
    logits = load_semantic_predictions(cache, semantic_checkpoint)
    objectness_threshold = float(semantic_checkpoint["objectness_threshold"])
    scene_order = sorted(
        range(len(cache["scene_id"])),
        key=lambda index: hashlib.sha256(str(cache["scene_id"][index]).encode()).hexdigest(),
    )[: args.max_scenes]
    scene_ids = [str(cache["scene_id"][index]) for index in scene_order]
    manifest = read_manifest(args.scene_manifest.resolve())
    missing = sorted(set(scene_ids) - set(manifest))
    if missing:
        raise ValueError(f"cohort scenes absent from manifest: {missing[:3]}")

    slots_by_scene = {
        str(cache["scene_id"][index]): selected_slots(
            cache, logits, index, objectness_threshold=objectness_threshold, top_k=args.top_k
        )
        for index in scene_order
    }
    labels_needed = sorted({label_id for slots in slots_by_scene.values() for slot in slots for label_id in slot["candidate_label_ids"]})
    prompts = {
        label_id: args.prompt_template.format(label=describe(labels[label_id]).replace("_", " "))
        for label_id in labels_needed
    }
    print(json.dumps({
        "cohort_frozen": True, "scenes": len(scene_ids),
        "predicted_slots": sum(len(value) for value in slots_by_scene.values()),
        "unique_global_prompts": len(prompts), "top_k": args.top_k,
    }, sort_keys=True), flush=True)

    pending = [scene_id for scene_id in scene_ids if not (fragment_dir / f"{scene_id}.pt").exists()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if pending:
        prompt_embeddings = encode_prompts(
            args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(),
            prompts.values(), batch_size=64,
        )
        detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
        if detector_payload.get("labels") != labels:
            raise ValueError("R1 detector and ontology labels differ")
        detector = load_model(len(labels), args.pretrainedsed_checkpoint_name, device, unfreeze_last_blocks=0)
        detector.load_state_dict(detector_payload["model_state_dict"], strict=True)
        detector.eval().requires_grad_(False)
        separator = _load_separator(args, device).eval().requires_grad_(False)
        started = time.time()
        calls = 0
        for position, scene_id in enumerate(pending, start=1):
            fragment = score_scene(
                scene_id=scene_id, row=manifest[scene_id], slots=slots_by_scene[scene_id],
                labels=labels, prompt_embeddings=prompt_embeddings, prompts=prompts,
                separator=separator, detector=detector, device=device,
                separator_batch_size=args.separator_batch_size,
            )
            _atomic_torch(fragment, fragment_dir / f"{scene_id}.pt")
            calls += int(fragment["separator_calls"])
            elapsed = time.time() - started
            rate = position / max(elapsed, 1e-6)
            print(
                f"scene={position}/{len(pending)} id={scene_id} calls={calls} "
                f"rate={rate:.3f}_scene_s eta={(len(pending)-position)/max(rate,1e-6)/60:.1f}_min",
                flush=True,
            )
        del detector, separator, prompt_embeddings, detector_payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fragments = {
        scene_id: torch.load(fragment_dir / f"{scene_id}.pt", map_location="cpu", weights_only=False)
        for scene_id in scene_ids
    }
    metrics = evaluate(cache, scene_order, fragments, iou_threshold=args.iou_threshold)
    baseline = metrics["methods"]["baseline_top1_label_id"]["top1_given_iou50_and_candidate_eligible_\u2191"]
    fusion = metrics["methods"]["fixed_fusion_top1_label_id"]["top1_given_iou50_and_candidate_eligible_\u2191"]
    gate = float(fusion) >= float(baseline) + 0.05
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "claim_boundary": "deterministic development pilot; not locked test",
        "cohort_selection": "first max_scenes after sorting SHA256(scene_id); frozen before inference",
        "qa_question_or_answer_used_as_input": False,
        "method": "top20_semantic_shortlist_then_frozen_audiosep_and_frozen_beats_r1_verification",
        "fixed_fusion": "z(slot_semantic_logit) + z(BEATs_R1_candidate_logit_on_AudioSep_stem_inside_slot)",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "cohort_scene_ids": scene_ids,
        "metrics": metrics,
        "success_gate": {"fixed_fusion_candidate_eligible_top1_improves_by_ge_0_05": gate},
        "decision": "scale_separator_verifier" if gate else "reject_audiosep_verifier_as_current_semantic_fix",
        "inputs": {
            "slot_cache_sha256": _sha256_file(cache_path),
            "semantic_checkpoint_sha256": _sha256_file(semantic_path),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
            "audiosep_checkpoint_sha256": _sha256_file(args.audiosep_checkpoint.resolve()),
            "scene_manifest_sha256": _sha256_file(args.scene_manifest.resolve()),
        },
        "separator_calls": sum(int(fragment["separator_calls"]) for fragment in fragments.values()),
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({
        "complete": True, "metrics": {key: value for key, value in metrics.items() if key != "items"},
        "success_gate": receipt["success_gate"], "decision": receipt["decision"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
