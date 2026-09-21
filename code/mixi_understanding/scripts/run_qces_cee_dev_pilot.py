#!/usr/bin/env python3
"""Run a schedule-matched CEE-off/on QCES-v5 held-out development pilot.

This is a validation-screening experiment, not a paper test result.  Execution
is fail-closed on both the microfit anti-collapse receipt and the AMP batch-3
memory-smoke receipt.  The two candidates share data, seed, initialization,
grouped batches, crops, frozen AudioSep, and checkpoint selection; only their
paired CEE loss weights differ.
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

from mixi_understanding.demo_contract import (  # noqa: E402
    DemoContractError,
    load_and_validate_health_receipt,
)
from mixi_understanding.qces.counterfactual import (  # noqa: E402
    COUNTERFACTUAL_METRIC_DIRECTIONS,
)
from mixi_understanding.scripts.run_qces_cee_memory_smoke import (  # noqa: E402
    CEE_WEIGHT_FLAGS,
    gpu_state,
    require_idle_gpu,
)
from mixi_understanding.scripts.run_qces_microfit_gate import (  # noqa: E402
    select_listening_item_ids,
)


FORMAT = "qces_cee_schedule_matched_dev_pilot_v1"
CANDIDATES = ("cee_off_matched", "cee_on")
EVAL_METRICS = {
    "evidence_sd_sdri_answerable": "maximize",
    "weakest_role_sd_sdr_answerable": "maximize",
    "answerable_temporal_iou": "maximize",
    "mean_no_evidence_retained_ratio": "minimize",
    "maximum_mixture_consistency_l1": "minimize",
}
PRIMARY_CEE_METRICS = (
    "surface_evidence_relative_l1",
    "family_temporal_delta_rmse",
    "family_transition_accuracy",
    "question_temporal_delta_rmse",
)
EXPECTED_SHA256 = {
    "train_manifest": "741c9b0d4285805bc879ba332f1c0278cf7105ed6a2ee59b79675d141064f252",
    "val_manifest": "4135f8ce757df30758a83f3d5f44592479778aecd49644c95b736d6df5a7b2d0",
    "foundation_train_receipt": "d21eabfed2bd2a3fdc1707181b8ad1ce6cbca81c18f82a0d7d1f94da54015d05",
    "foundation_val_receipt": "2efa1dccae2809057714e3ce9f16a1ed5cb75c888556b0b3fc62ebd48da0e2e2",
    "semantic_union_train": "459cd1cad625c63047d1e9c48a8c35db3cd7066e3e68ff4ba02a9e8328747f81",
    "semantic_union_val": "b2b6589ec9381eb657704d734fc9d8b43c34866475bf903d207c1b81fec500c9",
    "audiosep_config": "e7e2e1a089d1de5b58ee0ddeae978f5c8a4649ae0ddea724301363a1427f7f52",
    "audiosep_checkpoint": "37f1691fb067e2575f1ad1cfbfe44b7b3da18e52f33fcb2b0937b72952f11ba1",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_devpilot_train_seed2026.jsonl",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_devpilot_val_seed2026.jsonl",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_paper_devpilot_seed2026",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_cee_devpilot_seed2026",
    )
    parser.add_argument(
        "--microfit-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_microfit_seed2028/union_train_balanced_v3/checkpoint.pt",
    )
    parser.add_argument(
        "--microfit-health-receipt",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_microfit_seed2028/union_train_balanced_v3/demo_health_receipt.json",
    )
    parser.add_argument(
        "--cee-memory-receipt",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_cee_memory_smoke_seed10/amp_batch3/cee_memory_smoke_receipt.json",
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--listening-cases", type=int, default=4)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=9_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.max_steps <= 0:
        parser.error("epochs and max steps must be positive")
    if args.listening_cases < 0 or args.minimum_free_gpu_mib <= 0:
        parser.error("listening cases must be non-negative and GPU memory positive")
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


def artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "python": args.python,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "foundation_train": args.cache_root / "foundation_train",
        "foundation_val": args.cache_root / "foundation_val",
        "semantic_union_train": args.cache_root / "semantic_union_train.pt",
        "semantic_union_val": args.cache_root / "semantic_union_val.pt",
        "audiosep_root": args.audiosep_root,
        "audiosep_config": args.audiosep_config,
        "audiosep_checkpoint": args.audiosep_checkpoint,
    }


def validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    paths = artifact_paths(args)
    missing = [
        f"{name}: {path.resolve()}" for name, path in paths.items() if not path.exists()
    ]
    if missing:
        raise RuntimeError("missing CEE pilot artifact(s): " + "; ".join(missing))
    identities: dict[str, Any] = {}
    for name, path in paths.items():
        if path.is_file():
            identities[name] = _identity(path)
            expected = EXPECTED_SHA256.get(name)
            if expected is not None and identities[name]["sha256"] != expected:
                raise RuntimeError(
                    f"frozen CEE pilot hash mismatch for {name}: "
                    f"{identities[name]['sha256']} != {expected}"
                )
        else:
            receipt = path / "cache_receipt.json"
            if name.startswith("foundation") and not receipt.is_file():
                raise RuntimeError(f"foundation cache lacks receipt: {receipt}")
            identities[name] = {
                "path": str(path.resolve()),
                "cache_receipt": _identity(receipt) if receipt.is_file() else None,
            }
            expected = EXPECTED_SHA256.get(f"{name}_receipt")
            if (
                expected is not None
                and identities[name]["cache_receipt"]["sha256"] != expected
            ):
                raise RuntimeError(
                    f"frozen CEE pilot cache receipt mismatch for {name}"
                )
    selection_receipt = args.cache_root / "selection_receipt.json"
    if not selection_receipt.is_file():
        raise RuntimeError(
            f"missing identifier-only selection receipt: {selection_receipt}"
        )
    selection = json.loads(selection_receipt.read_text("utf-8"))
    selection_block = (
        selection.get("selection") if isinstance(selection, Mapping) else None
    )
    if (
        not isinstance(selection, Mapping)
        or not isinstance(selection_block, Mapping)
        or selection.get("paper_result_eligible") is not False
        or selection.get("test_split_accessed") is not False
        or selection_block.get("uses_identifiers_only") is not True
        or selection_block.get("uses_labels_answers_targets_or_metrics") is not False
    ):
        raise RuntimeError("CEE pilot selection receipt violates development scope")
    identities["selection_receipt"] = _identity(selection_receipt)
    return identities


def execution_gate_state(args: argparse.Namespace) -> dict[str, Any]:
    errors: list[str] = []
    microfit: Mapping[str, Any] | None = None
    memory: Mapping[str, Any] | None = None
    try:
        microfit = load_and_validate_health_receipt(
            args.microfit_checkpoint.resolve(),
            args.microfit_health_receipt.resolve(),
        )
    except (DemoContractError, OSError) as error:
        errors.append(f"microfit gate: {error}")
    try:
        payload = json.loads(args.cee_memory_receipt.resolve().read_text("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("receipt root is not an object")
        gates = payload.get("gates")
        if not isinstance(gates, Mapping) or gates.get("all_gates_passed") is not True:
            raise ValueError("all CEE memory gates did not pass")
        memory = payload
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"CEE memory gate: {error}")
    return {
        "ready": not errors,
        "errors": errors,
        "microfit_permission": microfit.get("permission") if microfit else None,
        "cee_memory_format": memory.get("format") if memory else None,
    }


def build_train_command(args: argparse.Namespace, candidate: str) -> list[str]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    output_dir = args.results_root.resolve() / f"{candidate}_train"
    command = [
        str(args.python.resolve()),
        str(CODE_ROOT / "mixi_understanding/scripts/train_qces.py"),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(args.train_manifest.resolve()),
        "--val-manifest",
        str(args.val_manifest.resolve()),
        "--backend",
        "audiosep",
        "--audiosep-root",
        str(args.audiosep_root.resolve()),
        "--audiosep-config",
        str(args.audiosep_config.resolve()),
        "--audiosep-checkpoint",
        str(args.audiosep_checkpoint.resolve()),
        "--foundation-feature-mode",
        "audiosep_clap",
        "--foundation-feature-cache",
        str((args.cache_root / "foundation_train").resolve()),
        "--val-foundation-feature-cache",
        str((args.cache_root / "foundation_val").resolve()),
        "--foundation-semantic-mixing-mode",
        "question_residual",
        "--temporal-role-mode",
        "independent_sigmoid",
        "--semantic-separation-mode",
        "union_single",
        "--semantic-targets",
        str((args.cache_root / "semantic_union_train.pt").resolve()),
        "--val-semantic-targets",
        str((args.cache_root / "semantic_union_val.pt").resolve()),
        "--semantic-weight",
        "2.0",
        "--weakest-role-weight",
        "0.25",
        "--role-relative-weight",
        "0.25",
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        "3",
        "--learning-rate",
        "0.0003",
        "--dropout",
        "0.1",
        "--crop-seconds",
        "10",
        "--num-workers",
        "0",
        "--log-every-epochs",
        "1",
        "--selection-metric",
        "evidence_sd_sdr",
        "--early-stopping-patience",
        "0",
        "--selection-min-delta",
        "0",
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--precision",
        "amp_fp16",
        "--no-separator-aware-refiner",
        "--save-every-epoch",
        "--deterministic",
    ]
    weight = "0" if candidate == "cee_off_matched" else "0.1"
    for flag in CEE_WEIGHT_FLAGS:
        command.extend((flag, weight))
    if candidate == "cee_off_matched":
        command.append("--force-counterfactual-batching")
    return command


def build_evaluate_command(
    args: argparse.Namespace, candidate: str, listening_ids: Sequence[str]
) -> list[str]:
    train_dir = args.results_root.resolve() / f"{candidate}_train"
    eval_dir = args.results_root.resolve() / f"{candidate}_eval"
    command = [
        str(args.python.resolve()),
        str(CODE_ROOT / "mixi_understanding/scripts/evaluate_qces.py"),
        "--checkpoint",
        str(train_dir / "checkpoint.pt"),
        "--audiosep-root",
        str(args.audiosep_root.resolve()),
        "--audiosep-config",
        str(args.audiosep_config.resolve()),
        "--audiosep-checkpoint",
        str(args.audiosep_checkpoint.resolve()),
        "--foundation-feature-cache",
        str((args.cache_root / "foundation_val").resolve()),
        "--manifest",
        str(args.val_manifest.resolve()),
        "--output-dir",
        str(eval_dir),
        "--batch-size",
        "1",
        "--device",
        "cuda",
    ]
    if listening_ids:
        for item_id in listening_ids:
            command.extend(("--render-item-id", item_id))
    else:
        command.append("--no-render-audio")
    return command


def _run(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=environment, check=True)


def candidate_execution_state(results_root: Path, candidate: str) -> dict[str, bool]:
    """Return resumable stage state while rejecting ambiguous partial outputs."""

    train_dir = results_root / f"{candidate}_train"
    eval_dir = results_root / f"{candidate}_eval"
    train_required = (train_dir / "checkpoint.pt", train_dir / "summary.json")
    eval_required = (eval_dir / "evaluation_report.json",)
    train_complete = all(path.is_file() for path in train_required)
    eval_complete = all(path.is_file() for path in eval_required)
    train_nonempty = train_dir.is_dir() and any(train_dir.iterdir())
    eval_nonempty = eval_dir.is_dir() and any(eval_dir.iterdir())
    if train_nonempty and not train_complete:
        raise RuntimeError(f"partial training output requires audit: {train_dir}")
    if eval_nonempty and not eval_complete:
        raise RuntimeError(f"partial evaluation output requires audit: {eval_dir}")
    if eval_complete and not train_complete:
        raise RuntimeError(f"evaluation exists without complete training: {eval_dir}")
    return {
        "train_complete": train_complete,
        "evaluation_complete": eval_complete,
    }


def _metric_improvement(direction: str, off: float, on: float) -> float:
    if direction == "maximize":
        return on - off
    if direction == "minimize":
        return off - on
    raise ValueError(f"unknown direction: {direction}")


def compare_candidates(results_root: Path) -> dict[str, Any]:
    summaries = {
        candidate: json.loads(
            (results_root / f"{candidate}_train/summary.json").read_text("utf-8")
        )
        for candidate in CANDIDATES
    }
    evaluations = {
        candidate: json.loads(
            (results_root / f"{candidate}_eval/evaluation_report.json").read_text(
                "utf-8"
            )
        )
        for candidate in CANDIDATES
    }
    off = summaries["cee_off_matched"]
    on = summaries["cee_on"]
    integrity_errors: list[str] = []
    if off.get("from_scratch_composer_initialization") != on.get(
        "from_scratch_composer_initialization"
    ):
        integrity_errors.append("composer initializations differ")
    off_cee = off.get("counterfactual_evidence_equivariance", {})
    on_cee = on.get("counterfactual_evidence_equivariance", {})
    if off_cee.get("train") != on_cee.get("train"):
        integrity_errors.append("group plans differ")
    if off.get("training_config", {}).get("seed") != on.get("training_config", {}).get(
        "seed"
    ):
        integrity_errors.append("training seeds differ")
    for candidate, expected_objective, expected_forced in (
        ("cee_off_matched", False, True),
        ("cee_on", True, False),
    ):
        training = summaries[candidate].get("training_config", {})
        cee = summaries[candidate].get("counterfactual_evidence_equivariance", {})
        if training.get("batch_size") != 3 or training.get("precision") != "amp_fp16":
            integrity_errors.append(f"{candidate} is not AMP batch-3")
        if cee.get("paired_objective_enabled") is not expected_objective:
            integrity_errors.append(f"{candidate} objective state mismatch")
        if cee.get("forced_schedule_matched_control") is not expected_forced:
            integrity_errors.append(f"{candidate} forced-control state mismatch")
    off_eval = evaluations["cee_off_matched"]["summary"]
    on_eval = evaluations["cee_on"]["summary"]
    waveform: dict[str, Any] = {}
    for metric, direction in EVAL_METRICS.items():
        off_value = float(off_eval[metric])
        on_value = float(on_eval[metric])
        arrow = "↑" if direction == "maximize" else "↓"
        waveform[f"{metric}_{arrow}"] = {
            "cee_off_matched": off_value,
            "cee_on": on_value,
            "cee_on_improvement_↑": _metric_improvement(direction, off_value, on_value),
        }
    off_val = off.get("best_validation_metrics") or {}
    on_val = on.get("best_validation_metrics") or {}
    paired: dict[str, Any] = {}
    for metric, metadata in COUNTERFACTUAL_METRIC_DIRECTIONS.items():
        if metric not in off_val or metric not in on_val:
            integrity_errors.append(f"missing paired validation metric: {metric}")
            continue
        direction = str(metadata["direction"])
        off_value = float(off_val[metric])
        on_value = float(on_val[metric])
        paired[str(metadata["display"])] = {
            "cee_off_matched": off_value,
            "cee_on": on_value,
            "cee_on_improvement_↑": _metric_improvement(direction, off_value, on_value),
        }
    primary_improvements = [
        paired[str(COUNTERFACTUAL_METRIC_DIRECTIONS[name]["display"])][
            "cee_on_improvement_↑"
        ]
        for name in PRIMARY_CEE_METRICS
        if str(COUNTERFACTUAL_METRIC_DIRECTIONS[name]["display"]) in paired
    ]
    primary_improved = sum(value > 0.0 for value in primary_improvements)
    on_health = {
        "evidence_sd_sdri_answerable_↑": float(on_eval["evidence_sd_sdri_answerable"]),
        "answerable_temporal_iou_↑": float(on_eval["answerable_temporal_iou"]),
        "mean_no_evidence_retained_ratio_↓": float(
            on_eval["mean_no_evidence_retained_ratio"]
        ),
        "maximum_mixture_consistency_l1_↓": float(
            on_eval["maximum_mixture_consistency_l1"]
        ),
    }
    anti_collapse = (
        on_health["evidence_sd_sdri_answerable_↑"] > 0.0
        and on_health["answerable_temporal_iou_↑"] >= 0.27
        and on_health["mean_no_evidence_retained_ratio_↓"] <= 0.10
        and on_health["maximum_mixture_consistency_l1_↓"] <= 1e-5
    )
    non_catastrophic = (
        float(on_eval["evidence_sd_sdr_answerable"])
        >= float(off_eval["evidence_sd_sdr_answerable"]) - 0.25
        and float(on_eval["weakest_role_sd_sdr_answerable"])
        >= float(off_eval["weakest_role_sd_sdr_answerable"]) - 0.25
        and float(on_eval["mean_no_evidence_retained_ratio"])
        <= min(0.10, float(off_eval["mean_no_evidence_retained_ratio"]) + 0.02)
    )
    promote = (
        not integrity_errors
        and anti_collapse
        and non_catastrophic
        and primary_improved >= 3
    )
    return {
        "format": FORMAT,
        "scope": "heldout_development_pilot_not_test_result",
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "integrity": {
            "all_passed": not integrity_errors,
            "errors": integrity_errors,
            "same_initialization_↑": not any(
                "initializations differ" in value for value in integrity_errors
            ),
            "same_grouped_schedule_design_↑": not any(
                "state mismatch" in value or "group plans differ" in value
                for value in integrity_errors
            ),
        },
        "waveform_metrics": waveform,
        "paired_validation_metrics": paired,
        "cee_on_health": on_health,
        "decision": {
            "anti_collapse_gate_passed": anti_collapse,
            "non_catastrophic_waveform_tradeoff_passed": non_catastrophic,
            "primary_cee_metrics_improved_count_↑": primary_improved,
            "primary_cee_metrics_required_↑": 3,
            "promote_cee_to_full_seeded_run": promote,
            "claim_boundary": (
                "one held-out development screen; promotion starts multi-seed "
                "validation and does not authorize any test claim"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    identities = validate_artifacts(args)
    listening_ids = select_listening_item_ids(args.val_manifest, args.listening_cases)
    gate_state = execution_gate_state(args)
    commands = {
        candidate: {
            "train": build_train_command(args, candidate),
            "evaluate": build_evaluate_command(args, candidate, listening_ids),
        }
        for candidate in CANDIDATES
    }
    plan = {
        "format": FORMAT,
        "purpose": "schedule_matched_CEE_validation_screen_not_test_result",
        "paper_result_eligible": False,
        "test_records_accessed_↓": 0,
        "execution_gates": gate_state,
        "artifacts": identities,
        "listening_item_ids": listening_ids,
        "commands": commands,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if not gate_state["ready"]:
        raise SystemExit("execution gates failed: " + "; ".join(gate_state["errors"]))
    comparison_path = args.results_root.resolve() / "comparison.json"
    if comparison_path.is_file():
        existing = json.loads(comparison_path.read_text("utf-8"))
        if not isinstance(existing, Mapping) or existing.get("format") != FORMAT:
            raise SystemExit(f"invalid existing comparison: {comparison_path}")
        print(json.dumps(existing, indent=2, sort_keys=True), flush=True)
        return
    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    for candidate in CANDIDATES:
        try:
            stage = candidate_execution_state(args.results_root.resolve(), candidate)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not stage["train_complete"]:
            _run(commands[candidate]["train"])
        if not stage["evaluation_complete"]:
            _run(commands[candidate]["evaluate"])
    comparison = compare_candidates(args.results_root.resolve())
    comparison["gpu_preflight"] = state
    comparison["plan"] = plan
    _atomic_json(comparison_path, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    if not comparison["integrity"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
