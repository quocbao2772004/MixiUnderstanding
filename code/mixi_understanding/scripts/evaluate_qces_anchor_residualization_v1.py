#!/usr/bin/env python3
"""Evaluate known-anchor subtraction for overlapping relational evidence.

The question exposes the anchor class (for example, ``the first bark``), but
not the answer class.  Frozen AudioSep is therefore called exactly once with
the anchor text.  Its estimated anchor is subtracted from the mixture and the
result is restricted to the answer-event window.  This tests whether removing
the known overlapping source makes the unknown answer acoustically cleaner.

Oracle time is used in v1 deliberately: this factorization isolates semantic
residualization from the already-measured temporal proposal error.  The answer
label is used only by an explicitly named upper bound, never by the proposed
anchor-residual condition.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.scripts.evaluate_audiosep_baselines import describe, encode_prompts


FORMAT = "qces_anchor_residualization_factorization_v1"
SAMPLE_RATE = 16_000
MODES = (
    "mixture__oracle_answer_window",
    "anchor_audiosep_residual__oracle_answer_window",
    "anchor_oracle_residual__oracle_answer_window",
    "answer_audiosep_oracle_semantic__oracle_answer_window",
)


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, default=data / "scene_manifest_test.jsonl")
    parser.add_argument("--qa-manifest", type=Path, default=data / "qa_manifest_test.jsonl")
    parser.add_argument("--dataset-root", type=Path, default=data)
    parser.add_argument("--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep")
    parser.add_argument(
        "--audiosep-config", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_anchor_residualization_overlap_stress_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Answerable diagnostic cap; 0 uses all 191.")
    parser.add_argument("--render-first", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row {line_number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {line_number}: {path}")
            rows.append(value)
    return rows


def resolve_under(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    root = root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes dataset root: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def read_mono(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz: {path}")
    return waveform.mean(1).astype(np.float32, copy=False)


def full_component(root: Path, event: Mapping[str, Any], samples: int) -> torch.Tensor:
    component = torch.from_numpy(read_mono(resolve_under(root, str(event["component_path"]))))
    start = int(event["placement_start_sample"])
    end = start + component.numel()
    if start < 0 or end > samples:
        raise ValueError(f"component placement outside mixture: {event['event_id']}")
    result = torch.zeros(samples, dtype=torch.float32)
    result[start:end] = component
    return result


def window_mask(samples: int, event: Mapping[str, Any]) -> torch.Tensor:
    start = max(0, min(samples, int(round(float(event["onset_seconds"]) * SAMPLE_RATE))))
    end = max(start, min(samples, int(round(float(event["offset_seconds"]) * SAMPLE_RATE))))
    result = torch.zeros(samples, dtype=torch.float32)
    result[start:end] = 1.0
    return result


def metric(candidate: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float().cpu()
    target = target.float().cpu()
    baseline = baseline.float().cpu()
    sd = float(scale_dependent_sdr(candidate[None], target[None])[0])
    base_sd = float(scale_dependent_sdr(baseline[None], target[None])[0])
    si = float(scale_invariant_sdr(candidate[None], target[None])[0])
    target_power = float(target.square().mean().clamp_min(1e-8))
    nmse = float((candidate - target).square().mean() / target_power)
    cosine = float(
        torch.dot(candidate, target)
        / (candidate.norm() * target.norm()).clamp_min(1e-8)
    )
    return {
        "sd_sdr_db_↑": sd,
        "sd_sdri_vs_mixture_window_db_↑": sd - base_sd,
        "si_sdr_db_↑": si,
        "normalized_mse_↓": nmse,
        "waveform_cosine_↑": cosine,
        "retained_energy_ratio": float(candidate.square().sum() / baseline.square().sum().clamp_min(1e-8)),
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        entry: dict[str, Any] = {"records": len(selected)}
        for key in (
            "sd_sdr_db_↑", "sd_sdri_vs_mixture_window_db_↑", "si_sdr_db_↑",
            "normalized_mse_↓", "waveform_cosine_↑", "retained_energy_ratio",
        ):
            values = [float(row["metrics"][key]) for row in selected]
            entry[f"mean_{key}"] = float(np.mean(values))
            entry[f"median_{key}"] = float(median(values))
        if mode != MODES[0]:
            entry["fraction_improved_sd_sdr_vs_mixture_↑"] = float(np.mean([
                row["metrics"]["sd_sdri_vs_mixture_window_db_↑"] > 0 for row in selected
            ]))
        result[mode] = entry
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_records < 0 or args.render_first < 0:
        raise SystemExit("invalid batch/max-records/render-first")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve()
    scene_rows = jsonl(args.scene_manifest)
    qa_rows = jsonl(args.qa_manifest)
    scenes = {str(row["scene_id"]): row for row in scene_rows}
    answerable = [row for row in qa_rows if not bool(row["no_evidence"])]
    if args.max_records:
        answerable = answerable[: args.max_records]
    if not answerable:
        raise ValueError("no answerable records selected")
    prompts = []
    records = []
    for qa in answerable:
        scene = scenes[str(qa["scene_id"])]
        events = {str(event["event_id"]): event for event in scene["events"]}
        anchor = events[str(qa["anchor_event_id"])]
        answer = events[str(qa["answer_event_id"])]
        anchor_prompt = describe(str(anchor["label"]).replace("_", " "))
        answer_prompt = describe(str(answer["label"]).replace("_", " "))
        prompts.extend((anchor_prompt, answer_prompt))
        records.append((qa, scene, anchor, answer, anchor_prompt, answer_prompt))

    embeddings = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), prompts, batch_size=64
    )
    gc.collect()
    device = torch.device(args.device)
    separator = AudioSepConditionedAdapter.from_repository(
        args.audiosep_root.resolve(), args.audiosep_config.resolve(),
        args.audiosep_checkpoint.resolve(), device=device, freeze_separator=True,
    ).ss_model.eval()
    item_rows: list[dict[str, Any]] = []
    rendered = 0
    with torch.inference_mode():
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start:batch_start + args.batch_size]
            mixtures = []
            anchor_conditions = []
            answer_conditions = []
            prepared = []
            for qa, scene, anchor, answer, anchor_prompt, answer_prompt in batch:
                mixture = torch.from_numpy(read_mono(resolve_under(root, str(scene["mixture_path"]))))
                samples = mixture.numel()
                target_anchor = full_component(root, anchor, samples)
                target_answer = full_component(root, answer, samples)
                # Component files are the exact additive render; fail closed if
                # their sum does not reproduce the stored mixture.
                reconstructed = sum(
                    (full_component(root, event, samples) for event in scene["events"]),
                    torch.zeros(samples),
                )
                if float((reconstructed - mixture).abs().max()) > 1e-6:
                    raise RuntimeError(f"component reconstruction mismatch: {scene['scene_id']}")
                mask = window_mask(samples, answer)
                mixtures.append(mixture)
                anchor_conditions.append(embeddings[anchor_prompt])
                answer_conditions.append(embeddings[answer_prompt])
                prepared.append((qa, scene, anchor, answer, target_anchor, target_answer, mask))
            mixture_batch = torch.stack(mixtures).to(device)
            anchor_condition = torch.stack(anchor_conditions).to(device)
            answer_condition = torch.stack(answer_conditions).to(device)
            anchor_estimate = separator(
                {"mixture": mixture_batch[:, None], "condition": anchor_condition}
            )["waveform"][:, 0]
            answer_estimate = separator(
                {"mixture": mixture_batch[:, None], "condition": answer_condition}
            )["waveform"][:, 0]
            for index, prepared_row in enumerate(prepared):
                qa, scene, anchor, answer, target_anchor, target_answer, mask = prepared_row
                mixture = mixture_batch[index].cpu()
                mask = mask.cpu()
                target_anchor = target_anchor.cpu()
                target_answer = target_answer.cpu()
                baseline = mixture * mask
                candidates = {
                    MODES[0]: baseline,
                    MODES[1]: (mixture - anchor_estimate[index].cpu()) * mask,
                    MODES[2]: (mixture - target_anchor) * mask,
                    MODES[3]: answer_estimate[index].cpu() * mask,
                }
                for mode, candidate in candidates.items():
                    values = metric(candidate, target_answer, baseline)
                    if not all(math.isfinite(value) for value in values.values()):
                        raise RuntimeError(f"non-finite metric: {scene['scene_id']} {mode}")
                    item_rows.append({
                        "item_id": qa["item_id"], "scene_id": scene["scene_id"],
                        "question": qa["question"], "anchor_label": anchor["label"],
                        "answer_label": answer["label"], "mode": mode, "metrics": values,
                        "overlap_fraction": scene["overlap_protocol"]["realized_target_pair_overlap_fraction"],
                        "other_over_anchor_snr_db": scene["overlap_protocol"]["measured_other_over_anchor_snr_db"],
                    })
                if rendered < args.render_first:
                    case = output / "audio" / str(scene["scene_id"])
                    case.mkdir(parents=True, exist_ok=True)
                    sf.write(case / "00_mixture_answer_window.wav", baseline.numpy(), SAMPLE_RATE)
                    sf.write(case / "01_anchor_audiosep_residual.wav", candidates[MODES[1]].numpy(), SAMPLE_RATE)
                    sf.write(case / "02_oracle_anchor_residual.wav", candidates[MODES[2]].numpy(), SAMPLE_RATE)
                    sf.write(case / "03_oracle_answer_audiosep.wav", candidates[MODES[3]].numpy(), SAMPLE_RATE)
                    sf.write(case / "04_target_answer.wav", target_answer.numpy(), SAMPLE_RATE)
                    (case / "metadata.json").write_text(
                        json.dumps({"question": qa["question"], "anchor": anchor["label"],
                                    "answer": answer["label"]}, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    rendered += 1
            print(f"evaluated {min(batch_start + len(batch), len(records))}/{len(records)}", flush=True)

    items_path = output / "items.jsonl"
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in item_rows),
        encoding="utf-8",
    )
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "protocol": {
            "records": len(records), "answerable_only": True,
            "oracle_time_for_factorization": True,
            "proposed_condition_uses_answer_label": False,
            "separator_calls_per_record_proposed": 1,
            "answer_audiosep_condition_is_oracle_upper_bound": True,
            "test_parameter_tuning": False,
        },
        "mode_registry": {
            MODES[0]: "Unseparated mixture restricted to the oracle answer window.",
            MODES[1]: "Proposed: X - AudioSep(X, known anchor text), restricted to the oracle answer window.",
            MODES[2]: "Oracle anchor-removal upper bound, restricted to the oracle answer window.",
            MODES[3]: "Oracle answer-label AudioSep upper bound, restricted to the oracle answer window.",
        },
        "summary": summarize(item_rows),
        "artifacts": {"items": str(items_path), "items_sha256": sha256(items_path), "rendered_cases": rendered},
        "inputs": {
            "scene_manifest": str(args.scene_manifest.resolve()), "scene_manifest_sha256": sha256(args.scene_manifest.resolve()),
            "qa_manifest": str(args.qa_manifest.resolve()), "qa_manifest_sha256": sha256(args.qa_manifest.resolve()),
            "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
            "audiosep_checkpoint_sha256": sha256(args.audiosep_checkpoint.resolve()),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
