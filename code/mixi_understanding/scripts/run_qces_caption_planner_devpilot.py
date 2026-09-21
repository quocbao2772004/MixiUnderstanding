#!/usr/bin/env python3
"""Run the frozen AF3-caption/planner -> AudioSep development comparator.

This runner is deliberately development-only.  It binds the 288-record QCES-v5
pilot, the frozen temporal controls, the local AF3 snapshot, and the official
AudioSep actuator.  Generation is resumable; an incomplete waveform evaluation
is rejected for manual audit instead of being overwritten implicitly.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.generate_qces_v5_caption_planner import (  # noqa: E402
    FORMAT_VERSION as PLANNER_FORMAT,
)
from mixi_understanding.scripts.evaluate_qces_v5_caption_planner_audiosep import (  # noqa: E402
    FORMAT_VERSION as EVALUATION_FORMAT,
)
from mixi_understanding.scripts.run_qces_cee_memory_smoke import (  # noqa: E402
    gpu_state,
    require_idle_gpu,
)


FORMAT = "qces_v5_caption_planner_devpilot_runner_v1"
EXPECTED_SHA256 = {
    "manifest": "4135f8ce757df30758a83f3d5f44592479778aecd49644c95b736d6df5a7b2d0",
    "temporal_report": "605080e1edba381245937598898b0979efc6a7a7e072aedc51341825d00c863d",
    "audiosep_config": "e7e2e1a089d1de5b58ee0ddeae978f5c8a4649ae0ddea724301363a1427f7f52",
    "audiosep_checkpoint": "37f1691fb067e2575f1ad1cfbfe44b7b3da18e52f33fcb2b0937b72952f11ba1",
}
EXPECTED_RECORDS = 288
EXPECTED_SCENES = 18
TOP_ENERGY_MODE = "top_energy_activity_frames__non_oracle"
ORACLE_TIME_MODE = "oracle_temporal_mask__upper_bound"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--generator-python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/qces-sam/bin/python"),
    )
    parser.add_argument(
        "--evaluator-python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_devpilot_val_seed2026.jsonl",
    )
    parser.add_argument(
        "--temporal-report",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_devpilot_temporal_baselines_seed2026/evaluation_report.json",
    )
    parser.add_argument(
        "--planner-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_caption_planner_devpilot_seed2026",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_caption_planner_audiosep_devpilot_seed2026",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/cuongpv/.cache/huggingface/hub/"
            "models--nvidia--audio-flamingo-3-hf/snapshots/"
            "7d4bae64ee29878af6504ae6f6bb3e40492838ad"
        ),
    )
    parser.add_argument(
        "--audiosep-root",
        type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep",
    )
    parser.add_argument(
        "--audiosep-config",
        type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=9_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    args = parser.parse_args(argv)
    if args.minimum_free_gpu_mib <= 0:
        parser.error("minimum free GPU memory must be positive")
    if not 0 <= args.maximum_gpu_utilization_percent <= 100:
        parser.error("maximum GPU utilization must lie in [0, 100]")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "generator_python": args.generator_python,
        "evaluator_python": args.evaluator_python,
        "manifest": args.manifest,
        "temporal_report": args.temporal_report,
        "model": args.model,
        "audiosep_root": args.audiosep_root,
        "audiosep_config": args.audiosep_config,
        "audiosep_checkpoint": args.audiosep_checkpoint,
    }
    missing = [
        f"{name}: {path.resolve()}" for name, path in paths.items() if not path.exists()
    ]
    if missing:
        raise RuntimeError("missing caption/planner artifact(s): " + "; ".join(missing))
    identities: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            identities[name] = _identity(path)
            expected = EXPECTED_SHA256.get(name)
            if expected is not None and identities[name]["sha256"] != expected:
                raise RuntimeError(
                    f"frozen caption/planner hash mismatch for {name}: "
                    f"{identities[name]['sha256']} != {expected}"
                )
        else:
            identities[name] = {"path": str(path.resolve()), "is_directory": True}
    manifest_name = args.manifest.name.casefold()
    if "test" in manifest_name or "devpilot" not in manifest_name:
        raise RuntimeError("runner accepts only the frozen development-pilot manifest")
    temporal = _read_json(args.temporal_report)
    if (
        temporal.get("format") != "qces_v5_temporal_only_baselines_v1"
        or temporal.get("selected_record_count") != EXPECTED_RECORDS
        or temporal.get("manifest_sha256") != EXPECTED_SHA256["manifest"]
    ):
        raise RuntimeError(
            "temporal baseline report violates the frozen pilot contract"
        )
    return identities


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    planner_report = args.planner_dir.resolve() / "planner_report.json"
    return {
        "generate": [
            str(args.generator_python.resolve()),
            str(
                CODE_ROOT
                / "mixi_understanding/scripts/generate_qces_v5_caption_planner.py"
            ),
            "--manifest",
            str(args.manifest.resolve()),
            "--output-dir",
            str(args.planner_dir.resolve()),
            "--model",
            str(args.model.resolve()),
            "--quantization",
            "4bit",
            "--dtype",
            "float16",
            "--device",
            "cuda",
            "--device-map",
            "cuda:0",
            "--attention-implementation",
            "sdpa",
            "--local-files-only",
            "--hash-model-weights",
            "--seed",
            str(args.seed),
        ],
        "evaluate": [
            str(args.evaluator_python.resolve()),
            str(
                CODE_ROOT
                / "mixi_understanding/scripts/evaluate_qces_v5_caption_planner_audiosep.py"
            ),
            "--manifest",
            str(args.manifest.resolve()),
            "--planner-report",
            str(planner_report),
            "--audiosep-root",
            str(args.audiosep_root.resolve()),
            "--audiosep-config",
            str(args.audiosep_config.resolve()),
            "--audiosep-checkpoint",
            str(args.audiosep_checkpoint.resolve()),
            "--output-dir",
            str(args.evaluation_dir.resolve()),
            "--device",
            "cuda",
            "--no-render-audio",
        ],
    }


def _validate_planner_report(path: Path) -> Mapping[str, Any]:
    payload = _read_json(path)
    summary = payload.get("summary")
    if (
        payload.get("format") != PLANNER_FORMAT
        or payload.get("manifest_sha256") != EXPECTED_SHA256["manifest"]
        or not isinstance(summary, Mapping)
        or summary.get("record_count_↑") != EXPECTED_RECORDS
        or summary.get("scene_count_↑") != EXPECTED_SCENES
        or summary.get("test_records_accessed_↓") != 0
    ):
        raise RuntimeError("caption/planner report violates the frozen pilot contract")
    return payload


def execution_state(args: argparse.Namespace) -> dict[str, bool]:
    planner_report = args.planner_dir.resolve() / "planner_report.json"
    evaluation_report = args.evaluation_dir.resolve() / "evaluation_report.json"
    planner_complete = planner_report.is_file()
    evaluation_complete = evaluation_report.is_file()
    if planner_complete:
        _validate_planner_report(planner_report)
    evaluation_nonempty = args.evaluation_dir.is_dir() and any(
        args.evaluation_dir.iterdir()
    )
    if evaluation_nonempty and not evaluation_complete:
        raise RuntimeError(
            f"partial waveform evaluation requires audit: {args.evaluation_dir.resolve()}"
        )
    if evaluation_complete and not planner_complete:
        raise RuntimeError("waveform evaluation exists without a planner report")
    return {
        "planner_complete": planner_complete,
        "evaluation_complete": evaluation_complete,
    }


def _run(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=environment, check=True)


def analyze_reports(
    planner_report: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    temporal_report: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    planner_summary = planner_report.get("summary")
    evaluation_summary = evaluation_report.get("summary")
    temporal_summaries = temporal_report.get("summaries")
    if not isinstance(planner_summary, Mapping):
        errors.append("planner summary missing")
        planner_summary = {}
    if not isinstance(evaluation_summary, Mapping):
        errors.append("waveform summary missing")
        evaluation_summary = {}
    if not isinstance(temporal_summaries, Mapping):
        errors.append("temporal summaries missing")
        temporal_summaries = {}
    top = temporal_summaries.get(TOP_ENERGY_MODE, {})
    oracle = temporal_summaries.get(ORACLE_TIME_MODE, {})
    if not isinstance(top, Mapping) or not isinstance(oracle, Mapping):
        errors.append("required temporal modes missing")
        top, oracle = {}, {}

    if evaluation_report.get("format") != EVALUATION_FORMAT:
        errors.append("waveform report format mismatch")
    if evaluation_report.get("manifest_sha256") != EXPECTED_SHA256["manifest"]:
        errors.append("waveform report manifest mismatch")
    if evaluation_report.get("selected_record_count") != EXPECTED_RECORDS:
        errors.append("waveform report record count mismatch")
    provenance = evaluation_report.get("planner_provenance")
    if not isinstance(provenance, Mapping) or provenance.get(
        "run_fingerprint"
    ) != planner_report.get("run_fingerprint"):
        errors.append("waveform report is not bound to the planner run")

    def number(source: Mapping[str, Any], key: str) -> float:
        value = source.get(key)
        if not isinstance(value, (int, float)):
            errors.append(f"missing numeric metric: {key}")
            return float("nan")
        return float(value)

    metrics = {
        "evidence_sd_sdri_answerable_mean_db_↑": (
            "evidence_sd_sdri_answerable_mean_db_↑"
        ),
        "weakest_role_sd_sdr_answerable_mean_db_↑": (
            "weakest_role_sd_sdr_answerable_mean_db_↑"
        ),
        "no_evidence_retained_ratio_mean_↓": "no_evidence_retained_ratio_mean_↓",
    }
    comparison: dict[str, Any] = {}
    improvements = 0
    for display, key in metrics.items():
        caption_value = number(evaluation_summary, key)
        top_value = number(top, key)
        oracle_value = number(oracle, key)
        minimize = display.endswith("↓")
        oriented_gain = (
            top_value - caption_value if minimize else caption_value - top_value
        )
        improvements += int(oriented_gain > 0.0)
        comparison[display] = {
            "caption_planner_audiosep": caption_value,
            "top_energy_activity": top_value,
            "oracle_temporal_upper_bound": oracle_value,
            "caption_over_top_energy_improvement_↑": oriented_gain,
        }

    reconstruction = number(
        evaluation_summary, "mixture_consistency_l1_sanity_maximum_↓"
    )
    if reconstruction > 1e-5:
        errors.append("mixture consistency sanity check failed")
    return {
        "format": FORMAT,
        "scope": "heldout_development_pilot_not_test_result",
        "paper_result_eligible": False,
        "test_records_accessed_↓": 0,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "integrity": {
            "all_passed": not errors,
            "errors": errors,
            "record_count_↑": evaluation_report.get("selected_record_count"),
            "scene_count_↑": planner_summary.get("scene_count_↑"),
            "mixture_consistency_l1_sanity_maximum_↓": reconstruction,
        },
        "planner_metrics": dict(planner_summary),
        "waveform_comparison": comparison,
        "development_interpretation": {
            "metrics_beating_top_energy_count_↑": improvements,
            "metrics_compared_count": len(metrics),
            "claim_boundary": (
                "strong non-oracle pipeline comparator only; no model promotion, "
                "test access, or paper claim is authorized"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = validate_artifacts(args)
    commands = build_commands(args)
    try:
        stages = execution_state(args)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    plan = {
        "format": FORMAT,
        "purpose": "strong_non_oracle_caption_planner_audiosep_devpilot_comparator",
        "paper_result_eligible": False,
        "test_records_accessed_↓": 0,
        "artifacts": artifacts,
        "stages": stages,
        "commands": commands,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return

    receipt_path = args.evaluation_dir.resolve() / "devpilot_comparison_receipt.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if receipt.get("format") != FORMAT:
            raise SystemExit(f"invalid existing receipt: {receipt_path}")
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
        return

    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    if not stages["planner_complete"]:
        _run(commands["generate"])
    planner_path = args.planner_dir.resolve() / "planner_report.json"
    planner = _validate_planner_report(planner_path)
    if not stages["evaluation_complete"]:
        _run(commands["evaluate"])
    evaluation_path = args.evaluation_dir.resolve() / "evaluation_report.json"
    evaluation = _read_json(evaluation_path)
    temporal = _read_json(args.temporal_report)
    receipt = analyze_reports(planner, evaluation, temporal)
    receipt.update(
        {
            "gpu_preflight": state,
            "artifacts": artifacts,
            "planner_report": _identity(planner_path),
            "evaluation_report": _identity(evaluation_path),
            "temporal_report": _identity(args.temporal_report),
        }
    )
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not receipt["integrity"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
