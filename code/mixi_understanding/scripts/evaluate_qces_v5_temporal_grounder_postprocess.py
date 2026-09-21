#!/usr/bin/env python3
"""Calibrate temporal-grounder gate post-processing on train and evaluate val."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.signal import qces_linear_interpolate_1d
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _v5_item_metrics,
    summarize_v5_items,
    write_item_artifacts,
)
from mixi_understanding.scripts.evaluate_qces_v5_structured_planner_audiosep import (
    MODE_PLANNER_TEXT_GATE,
    MODE_PLANNER_TEXT_NO_GATE,
    metadata as planner_metadata,
    plan_record,
    planned_gate,
)
from mixi_understanding.scripts.train_qces_v5_temporal_grounder_audiosep import (
    MODE_HARD_GATE,
    MODE_SOFT_GATE,
    TemporalGrounder,
    apply_normalization,
    build_or_load_audiosep_cache,
    frame_metrics,
    make_feature_packs,
    run_grounder,
    select_records,
)


FORMAT_VERSION = "qces_v5_temporal_grounder_postprocess_v1"
MODE_CALIBRATED_GATE = "temporal_grounder__train_calibrated_gate"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--render-item-id", action="append", default=[])
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args(argv)
    if len(args.render_item_id) != len(set(args.render_item_id)):
        parser.error("--render-item-id values must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TemporalGrounder, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu")
    model = TemporalGrounder(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for gamma in (0.20, 0.35, 0.50, 0.70, 1.00):
        candidates.append({"kind": "soft_power", "gamma": gamma})
    for scale in (0.25, 0.35, 0.50, 0.65, 0.80):
        candidates.append({"kind": "soft_rescale", "scale": scale})
    for threshold in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        for dilation in (1, 3, 5, 9, 15, 25):
            candidates.append(
                {
                    "kind": "hard_threshold_dilate",
                    "threshold": threshold,
                    "dilation_frames": dilation,
                }
            )
    return candidates


def candidate_name(candidate: Mapping[str, Any]) -> str:
    if candidate["kind"] == "soft_power":
        return f"soft_power_gamma_{candidate['gamma']}"
    if candidate["kind"] == "soft_rescale":
        return f"soft_rescale_scale_{candidate['scale']}"
    if candidate["kind"] == "hard_threshold_dilate":
        return (
            f"hard_t_{candidate['threshold']}"
            f"_dilate_{candidate['dilation_frames']}"
        )
    return str(candidate)


def postprocess_frames(probability: torch.Tensor, candidate: Mapping[str, Any]) -> torch.Tensor:
    kind = candidate["kind"]
    if kind == "soft_power":
        return probability.clamp(0.0, 1.0).pow(float(candidate["gamma"]))
    if kind == "soft_rescale":
        return (probability / float(candidate["scale"])).clamp(0.0, 1.0)
    if kind == "hard_threshold_dilate":
        gate = (probability >= float(candidate["threshold"])).float()
        dilation = int(candidate["dilation_frames"])
        if dilation > 1:
            if dilation % 2 == 0:
                dilation += 1
            gate = F.max_pool1d(
                gate[None, None], kernel_size=dilation, stride=1, padding=dilation // 2
            )[0, 0]
        return gate
    raise ValueError(f"unknown candidate kind: {kind!r}")


def frames_to_samples(gate_frames: torch.Tensor, samples: int) -> torch.Tensor:
    return qces_linear_interpolate_1d(gate_frames[None, None].float(), samples)[0, 0].clamp(0.0, 1.0)


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


@dataclass(frozen=True)
class WavePack:
    record: QCESV5Record
    plan: Any
    mixture: torch.Tensor
    target: torch.Tensor
    target_residual: torch.Tensor
    raw: torch.Tensor
    target_frames: torch.Tensor


def prepare_wave_packs(
    *,
    dataset: QCESManifestDataset,
    records: Sequence[QCESV5Record],
    cache: Mapping[str, Any],
    frames: int,
) -> list[WavePack]:
    raw_by_id = cache["raw_audiosep_by_id"]
    packs: list[WavePack] = []
    for index, record in enumerate(records):
        example = dataset[index]
        target_frames = F.adaptive_max_pool1d(
            ((example.anchor_mask + example.answer_mask).clamp_max(1.0))[None, None],
            frames,
        )[0, 0]
        packs.append(
            WavePack(
                record=record,
                plan=plan_record(record),
                mixture=example.mixture,
                target=example.evidence,
                target_residual=example.residual,
                raw=raw_by_id[record.sample_id].float(),
                target_frames=target_frames,
            )
        )
    return packs


def evaluate_candidate(
    *,
    packs: Sequence[WavePack],
    probabilities: Mapping[str, torch.Tensor],
    candidate: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sd_sdri: list[float] = []
    si_sdri: list[float] = []
    l1: list[float] = []
    frame_f1: list[float] = []
    frame_iou: list[float] = []
    retained: list[float] = []
    for pack in packs:
        record = pack.record
        mixture = pack.mixture.to(device)
        target = pack.target.to(device)
        target_residual = pack.target_residual.to(device)
        raw = pack.raw.to(device)
        if pack.plan.no_evidence:
            evidence = torch.zeros_like(mixture)
            gate_frames = torch.zeros_like(probabilities[record.sample_id])
        else:
            gate_frames = postprocess_frames(probabilities[record.sample_id], candidate)
            gate = frames_to_samples(gate_frames.to(device), mixture.numel())
            evidence = raw * gate
        metrics, descriptives = _v5_item_metrics(
            no_evidence=record.no_evidence,
            evidence=evidence,
            mixture=mixture,
            target=target,
            target_residual=target_residual,
        )
        frame = frame_metrics(gate_frames.cpu(), pack.target_frames.cpu(), 0.5)
        if not record.no_evidence:
            sd_sdri.append(float(metrics["evidence_sd_sdri_db_↑"]))
            si_sdri.append(float(metrics["evidence_si_sdri_db_↑"]))
        l1.append(float(metrics["evidence_l1_↓"]))
        frame_f1.append(float(frame["frame_f1_↑"]))
        frame_iou.append(float(frame["frame_iou_↑"]))
        retained.append(float(descriptives["evidence_retained_ratio"]))
    return {
        "candidate": dict(candidate),
        "candidate_name": candidate_name(candidate),
        "evidence_sd_sdri_answerable_mean_db_↑": mean(sd_sdri),
        "evidence_si_sdri_answerable_mean_db_↑": mean(si_sdri),
        "evidence_l1_mean_↓": mean(l1),
        "frame_f1_mean_↑": mean(frame_f1),
        "frame_iou_mean_↑": mean(frame_iou),
        "retained_ratio_mean": mean(retained),
    }


def evaluate_best_items(
    *,
    args: argparse.Namespace,
    packs: Sequence[WavePack],
    probabilities: Mapping[str, torch.Tensor],
    candidate: Mapping[str, Any],
    output_dir: Path,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    render_ids = set(args.render_item_id)
    items: list[dict[str, Any]] = []
    for pack in packs:
        record = pack.record
        plan = pack.plan
        mixture = pack.mixture.to(device)
        target = pack.target.to(device)
        target_residual = pack.target_residual.to(device)
        raw = pack.raw.to(device)
        planned = planned_gate(record, plan, mixture.numel(), device, 0.0)
        probability = probabilities[record.sample_id]
        soft_gate = frames_to_samples(probability.to(device), mixture.numel())
        hard_gate = (soft_gate >= float(args.checkpoint_threshold)).float() if hasattr(args, "checkpoint_threshold") else (soft_gate >= 0.5).float()
        calibrated_frames = postprocess_frames(probability, candidate)
        calibrated_gate = frames_to_samples(calibrated_frames.to(device), mixture.numel())
        mode_to_evidence = {
            MODE_PLANNER_TEXT_NO_GATE: raw,
            MODE_PLANNER_TEXT_GATE: raw * planned,
            MODE_SOFT_GATE: raw * soft_gate,
            MODE_HARD_GATE: raw * hard_gate,
            MODE_CALIBRATED_GATE: raw * calibrated_gate,
        }
        if plan.no_evidence:
            mode_to_evidence = {
                mode: torch.zeros_like(mixture) for mode in mode_to_evidence
            }
        frame = frame_metrics(calibrated_frames.cpu(), pack.target_frames.cpu(), 0.5)
        for mode, evidence in mode_to_evidence.items():
            residual = mixture - evidence
            metrics, descriptives = _v5_item_metrics(
                no_evidence=record.no_evidence,
                evidence=evidence,
                mixture=mixture,
                target=target,
                target_residual=target_residual,
            )
            item = {
                **planner_metadata(record, plan),
                "split_evaluated": split,
                "mode": mode,
                "prompt": plan.prompt,
                "calibration_candidate": dict(candidate),
                "frame_metrics": frame,
                "metrics": metrics,
                "descriptives": descriptives,
            }
            should_render = not args.no_render_audio and (
                not render_ids or record.sample_id in render_ids
            )
            write_item_artifacts(
                question_dir=(
                    output_dir
                    / split
                    / mode
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                ),
                item=item,
                evidence=evidence,
                residual=residual,
                sample_rate=record.sample_rate,
                render_audio=should_render,
            )
            items.append(item)
    return items, summarize_by_mode(items)


def summarize_by_mode(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in sorted({str(item["mode"]) for item in items}):
        subset = [item for item in items if str(item["mode"]) == mode]
        summary = summarize_v5_items(subset)
        for key in ("frame_iou_↑", "frame_precision_↑", "frame_recall_↑", "frame_f1_↑"):
            values = [float(item["frame_metrics"][key]) for item in subset]
            summary[f"{key[:-1]}mean_↑"] = mean(values)
        result[mode] = summary
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.checkpoint, device)
    args.checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    frames = int(checkpoint["frames"])
    train_dataset = QCESManifestDataset(args.train_manifest.resolve(), crop_samples=None)
    val_dataset = QCESManifestDataset(args.val_manifest.resolve(), crop_samples=None)
    train_records = select_records(train_dataset, 0)
    val_records = select_records(val_dataset, 0)
    train_cache = build_or_load_audiosep_cache(
        args=args,
        split="train",
        dataset=train_dataset,
        records=train_records,
        device=device,
    )
    val_cache = build_or_load_audiosep_cache(
        args=args,
        split="val",
        dataset=val_dataset,
        records=val_records,
        device=device,
    )
    train_packs = make_feature_packs(
        dataset=train_dataset, records=train_records, cache=train_cache, frames=frames
    )
    val_packs = make_feature_packs(
        dataset=val_dataset, records=val_records, cache=val_cache, frames=frames
    )
    train_packs = apply_normalization(train_packs, checkpoint["feature_mean"], checkpoint["feature_std"])
    val_packs = apply_normalization(val_packs, checkpoint["feature_mean"], checkpoint["feature_std"])
    train_prob = run_grounder(model, train_packs, device, args.batch_size)
    val_prob = run_grounder(model, val_packs, device, args.batch_size)
    train_wave_packs = prepare_wave_packs(
        dataset=train_dataset,
        records=train_records,
        cache=train_cache,
        frames=frames,
    )
    val_wave_packs = prepare_wave_packs(
        dataset=val_dataset,
        records=val_records,
        cache=val_cache,
        frames=frames,
    )
    candidate_results = [
        evaluate_candidate(
            packs=train_wave_packs,
            probabilities=train_prob,
            candidate=candidate,
            device=device,
        )
        for candidate in candidate_grid()
    ]
    candidate_results.sort(
        key=lambda row: (
            float(row["evidence_sd_sdri_answerable_mean_db_↑"]),
            float(row["frame_f1_mean_↑"]),
        ),
        reverse=True,
    )
    best_candidate = candidate_results[0]["candidate"]
    train_items, train_summary = evaluate_best_items(
        args=args,
        packs=train_wave_packs,
        probabilities=train_prob,
        candidate=best_candidate,
        output_dir=output_dir,
        split="train",
        device=device,
    )
    val_items, val_summary = evaluate_best_items(
        args=args,
        packs=val_wave_packs,
        probabilities=val_prob,
        candidate=best_candidate,
        output_dir=output_dir,
        split="val",
        device=device,
    )
    report = {
        "format": FORMAT_VERSION,
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "val_manifest_sha256": sha256_file(args.val_manifest.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "frames": frames,
        "selected_candidate": best_candidate,
        "selected_candidate_name": candidate_name(best_candidate),
        "selection_metric": "train evidence_sd_sdri_answerable_mean_db_↑",
        "candidate_train_sweep": candidate_results,
        "summaries_by_split": {
            "train": train_summary,
            "val": val_summary,
        },
        "protocol_limitations": [
            "Post-processing candidate is selected on train waveform metrics only.",
            "Planner/prompt still uses annotation-provided event inventory.",
        ],
        "items": {
            "train": train_items,
            "val": val_items,
        },
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summaries_by_split"]["val"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
