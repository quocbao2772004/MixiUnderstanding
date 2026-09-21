#!/usr/bin/env python3
"""Run the predeclared Phi-4 MM cross-family oracle calibration pilot.

The 24-record development subset is selected from six held-out scene families
using only family ID, base-variant ID, and fixed question slots 0/3/8/14.  It
contains no metric-driven selection.  Execution requires the independent CUDA
forward/repeatability receipt; dry-run freezes the subset and validates every
condition without loading Phi-4 MM.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.evaluate_qces_audioqa import (  # noqa: E402
    FORMAT_VERSION as AUDIOQA_FORMAT,
    OPTION_ORDER_CONTROL_SUFFIX,
)
from mixi_understanding.scripts.run_qces_phi4mm_forward_gate import (  # noqa: E402
    FORMAT as FORWARD_FORMAT,
    MODEL_ID,
    REVISION,
    _identity,
    command_environment,
    gpu_state,
    load_monitor,
    require_idle_gpu,
    run_monitored,
)

FORMAT = "qces_phi4mm_cross_family_oracle_calibration_v1"
SOURCE_MANIFEST_SHA256 = (
    "4135f8ce757df30758a83f3d5f44592479778aecd49644c95b736d6df5a7b2d0"
)
SELECTED_IDS_SHA256 = "103f5630f610d6f9cbfaedc8b1ee84e9ff37a024e1fb747f819d0b1ad91aa53b"
QUESTION_SLOTS = (0, 3, 8, 14)
BASE_CONDITIONS = (
    "mixture",
    "oracle_evidence",
    "oracle_residual",
    "shuffled_oracle_evidence",
    "question_only",
)
OPTION_ORDER_CONDITIONS = ("mixture", "oracle_evidence", "question_only")
EXPECTED_CONDITIONS = BASE_CONDITIONS + tuple(
    condition + OPTION_ORDER_CONTROL_SUFFIX for condition in OPTION_ORDER_CONDITIONS
)
LOG_SCORE_EFFECTS = (
    "oracle_evidence_gold_log_score_gain_over_question_only_↑",
    "oracle_evidence_gold_log_score_gain_over_shuffled_oracle_evidence_↑",
    "oracle_evidence_gold_log_score_gain_over_oracle_residual_↑",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/qces-phi4mm/bin/python"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_devpilot_val_seed2026.jsonl",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_phi4mm_oracle_calibration_seed2026",
    )
    parser.add_argument(
        "--forward-receipt",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_phi4mm_forward_gate_seed2026/"
        "phi4mm_forward_gate_receipt.json",
    )
    parser.add_argument(
        "--attention-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=9_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7_200)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0 or args.timeout_seconds <= 0:
        parser.error("bootstrap samples and timeout must be positive")
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


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def select_subset(source_manifest: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = source_manifest.resolve()
    if not source.is_file() or _sha256(source) != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("Phi calibration source manifest hash mismatch")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid source row at {source}:{line_number}")
            rows.append(row)
    family_ids = sorted(
        {str(row["scene_family_id"]) for row in rows if "scene_family_id" in row}
    )
    if len(family_ids) != 6:
        raise RuntimeError(f"expected six pilot families, found {len(family_ids)}")
    selected: list[dict[str, Any]] = []
    for family_id in family_ids:
        for question_index in QUESTION_SLOTS:
            matches = [
                row
                for row in rows
                if row.get("scene_family_id") == family_id
                and row.get("variant_id") == "base"
                and row.get("question_index") == question_index
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "identifier-only Phi selection is not unique for "
                    f"{family_id}/q{question_index}: {len(matches)}"
                )
            selected.append(matches[0])
    selected_ids = [str(row["id"]) for row in selected]
    if _sha256_json(selected_ids) != SELECTED_IDS_SHA256:
        raise RuntimeError("frozen Phi calibration item IDs changed")
    receipt = {
        "format": "qces_phi4mm_identifier_only_selection_v1",
        "source_manifest": _identity(source),
        "selection": {
            "scene_family_ids": family_ids,
            "variant_id": "base",
            "question_slots": list(QUESTION_SLOTS),
            "uses_identifiers_only": True,
            "uses_audio_labels_answers_targets_or_metrics": False,
            "selected_records_↑": len(selected),
            "selected_scene_families_↑": len(family_ids),
            "selected_item_ids": selected_ids,
            "selected_item_ids_sha256": SELECTED_IDS_SHA256,
        },
        "development_only": True,
        "test_split_accessed": False,
    }
    return selected, receipt


def write_subset(root: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = root.resolve() / "oracle_calibration_subset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"existing Phi calibration subset changed: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    return path


def build_command(
    args: argparse.Namespace,
    subset_manifest: Path,
    output_dir: Path,
    *,
    validate_only: bool = False,
) -> list[str]:
    command = [
        str(args.python.resolve()),
        str(CODE_ROOT / "mixi_understanding/scripts/evaluate_qces_audioqa.py"),
        "--auditor",
        "phi4mm",
        "--manifest",
        str(subset_manifest.resolve()),
        "--dataset-root",
        str(args.source_manifest.resolve().parent),
        "--output-dir",
        str(output_dir.resolve()),
        "--model",
        MODEL_ID,
        "--revision",
        REVISION,
        "--conditions",
        *BASE_CONDITIONS,
        "--option-order-control-conditions",
        *OPTION_ORDER_CONDITIONS,
        "--split",
        "val",
        "--quantization",
        "4bit",
        "--dtype",
        "float16",
        "--device",
        "cuda",
        "--device-map",
        "auto",
        "--attention-implementation",
        args.attention_implementation,
        "--local-files-only",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    if validate_only:
        command.append("--validate-only")
    return command


def _run_validate_only(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=command_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Phi calibration validate-only failed: "
            + (completed.stderr or completed.stdout)[-4_000:]
        )
    payload = json.loads(completed.stdout)
    if (
        not isinstance(payload, dict)
        or payload.get("records") != 24
        or payload.get("record_conditions") != 24 * len(EXPECTED_CONDITIONS)
    ):
        raise RuntimeError(f"unexpected calibration validation counts: {payload}")
    return payload


def _current_identity_matches(value: Any) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
        return False
    path = Path(value["path"])
    return path.is_file() and _identity(path) == value


def validate_forward_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path.resolve())
    if (
        receipt.get("format") != FORWARD_FORMAT
        or receipt.get("all_gates_passed") is not True
        or receipt.get("permission") != "phi4mm_oracle_calibration_authorized"
        or not isinstance(receipt.get("repeatability"), Mapping)
        or receipt["repeatability"].get("all_checks_passed") is not True
    ):
        raise RuntimeError("Phi forward receipt does not authorize calibration")
    for stage in ("primary", "repeat"):
        block = receipt.get(stage)
        if not isinstance(block, Mapping):
            raise RuntimeError(f"Phi forward receipt lacks {stage} output")
        for key in ("metadata", "report", "items"):
            if not _current_identity_matches(block.get(key)):
                raise RuntimeError(f"Phi forward {stage}/{key} artifact changed")
    assets = receipt.get("assets")
    snapshot_files = (
        assets.get("snapshot_files") if isinstance(assets, Mapping) else None
    )
    if not isinstance(snapshot_files, Mapping):
        raise RuntimeError("Phi forward receipt lacks snapshot identities")
    for relative, identity in snapshot_files.items():
        if not _current_identity_matches(identity):
            raise RuntimeError(f"Phi snapshot artifact changed: {relative}")
    return receipt


def validate_calibration_output(
    args: argparse.Namespace, subset: Path, output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = output / "run_metadata.json"
    report_path = output / "evaluation_report.json"
    items_path = output / "items.jsonl"
    for path in (metadata_path, report_path, items_path):
        if not path.is_file():
            raise RuntimeError(f"Phi calibration output is incomplete: {path}")
    metadata = _read_json(metadata_path)
    report = _read_json(report_path)
    run_config = metadata.get("run_config")
    runtime = metadata.get("runtime_model")
    if not isinstance(run_config, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("Phi calibration lacks run/runtime metadata")
    evaluator = CODE_ROOT / "mixi_understanding/scripts/evaluate_qces_audioqa.py"
    checks = {
        "report_metadata_fingerprint_match": report.get("run_fingerprint")
        == metadata.get("run_fingerprint"),
        "current_evaluator_script_match": metadata.get("script_sha256")
        == _sha256(evaluator),
        "run_config_evaluator_match": run_config.get("evaluator_script_sha256")
        == _sha256(evaluator),
        "manifest_path_match": run_config.get("manifest") == str(subset.resolve()),
        "manifest_hash_match": run_config.get("manifest_sha256") == _sha256(subset),
        "dataset_root_match": run_config.get("dataset_root")
        == str(args.source_manifest.resolve().parent),
        "conditions_match": run_config.get("conditions") == list(EXPECTED_CONDITIONS),
        "auditor_match": run_config.get("auditor") == "phi4mm",
        "model_match": run_config.get("model") == MODEL_ID,
        "revision_match": run_config.get("revision") == REVISION,
        "quantization_match": run_config.get("quantization") == "4bit",
        "resolved_commit_match": runtime.get("resolved_commit_hash") == REVISION,
        "auditor_architecture_match": runtime.get("auditor_family")
        == "phi4_multimodal",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"Phi calibration provenance failed: {failed}")
    return metadata, report


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"Phi calibration lacks numeric metric: {name}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Phi calibration metric is non-finite: {name}")
    return result


def analyze_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("format") != AUDIOQA_FORMAT:
        raise RuntimeError(f"wrong AudioQA report format: {report.get('format')}")
    counts = report.get("counts")
    paired = report.get("paired_metrics")
    intervals = report.get("paired_scene_bootstrap_95ci")
    conditions = report.get("condition_metrics")
    option = report.get("option_order_control_metrics_by_condition")
    subsets = report.get("paired_subset_counts")
    if not all(
        isinstance(value, Mapping)
        for value in (counts, paired, intervals, conditions, option, subsets)
    ):
        raise RuntimeError("Phi calibration report is structurally incomplete")
    integrity = {
        "records_equal_24_↑": counts.get("records") == 24,
        "answerable_equal_18_↑": counts.get("answerable") == 18,
        "no_evidence_equal_6_↑": counts.get("no_evidence") == 6,
        "conditions_equal_8_↑": counts.get("conditions") == len(EXPECTED_CONDITIONS),
        "record_conditions_equal_192_↑": counts.get("completed_record_conditions")
        == 24 * len(EXPECTED_CONDITIONS),
    }
    point_effects = {name: _finite_metric(paired, name) for name in LOG_SCORE_EFFECTS}
    ci_lower: dict[str, float] = {}
    for name in LOG_SCORE_EFFECTS:
        interval = intervals.get(name)
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)
        ):
            raise RuntimeError(f"Phi calibration lacks bootstrap CI: {name}")
        ci_lower[name] = float(interval[0])

    oracle = conditions.get("oracle_evidence")
    if not isinstance(oracle, Mapping):
        raise RuntimeError("Phi calibration lacks oracle-E condition metrics")
    oracle_accuracy = _finite_metric(oracle, "answerable_accuracy_↑")
    chance = _finite_metric(oracle, "answerable_candidate_aware_chance_accuracy_↑")
    no_evidence_accuracy = _finite_metric(oracle, "no_evidence_accuracy_↑")
    audio_dependent = subsets.get(
        "audio_dependent_mixture_correct_question_only_wrong_records"
    )
    if isinstance(audio_dependent, bool) or not isinstance(audio_dependent, int):
        raise RuntimeError("Phi calibration lacks audio-dependent subset count")

    option_checks: dict[str, bool] = {}
    option_values: dict[str, dict[str, float]] = {}
    for condition in OPTION_ORDER_CONDITIONS:
        metrics = option.get(condition)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"Phi calibration lacks option control: {condition}")
        position = _finite_metric(metrics, "option_order_gold_position_changed_rate_↑")
        invariance = _finite_metric(
            metrics, "option_order_semantic_prediction_invariance_↑"
        )
        accuracy_gap = _finite_metric(metrics, "option_order_accuracy_absolute_gap_↓")
        option_values[condition] = {
            "gold_position_changed_rate_↑": position,
            "semantic_prediction_invariance_↑": invariance,
            "accuracy_absolute_gap_↓": accuracy_gap,
        }
        option_checks[f"{condition}_gold_position_changed_rate_equal_1_↑"] = (
            math.isclose(position, 1.0, abs_tol=1e-12)
        )
        option_checks[f"{condition}_semantic_invariance_at_least_0.75_↑"] = (
            invariance >= 0.75
        )
        option_checks[f"{condition}_accuracy_gap_at_most_0.15_↓"] = accuracy_gap <= 0.15

    mixture_audio_gain = _finite_metric(
        paired, "mixture_gold_log_score_gain_over_question_only_↑"
    )
    pilot_checks = {
        **integrity,
        **option_checks,
        "all_oracle_log_score_effects_positive_↑": all(
            value > 0.0 for value in point_effects.values()
        ),
        "oracle_answerable_accuracy_margin_at_least_0.05_↑": oracle_accuracy
        >= chance + 0.05,
        "oracle_no_evidence_accuracy_at_least_0.50_↑": no_evidence_accuracy >= 0.50,
        "mixture_gold_log_score_gain_over_question_only_positive_↑": (
            mixture_audio_gain > 0.0
        ),
        "audio_dependent_records_at_least_3_↑": audio_dependent >= 3,
    }
    pilot_go = all(pilot_checks.values())
    confirmatory_checks = {
        f"{name}_bootstrap_lower_above_zero_↑": lower > 0.0
        for name, lower in ci_lower.items()
    }
    confirmatory_pass = pilot_go and all(confirmatory_checks.values())
    if confirmatory_pass:
        decision = "phi4mm_predicted_dev_pilot_authorized"
    elif pilot_go:
        decision = "phi4mm_oracle_expansion_required"
    else:
        decision = "phi4mm_auditor_rejected_on_pilot"
    return {
        "pilot_go": pilot_go,
        "confirmatory_pass": confirmatory_pass,
        "decision": decision,
        "pilot_checks": pilot_checks,
        "confirmatory_checks": confirmatory_checks,
        "oracle_log_score_effects_↑": point_effects,
        "oracle_log_score_effect_bootstrap_lower_95ci_↑": ci_lower,
        "oracle_answerable_accuracy_↑": oracle_accuracy,
        "candidate_aware_chance_accuracy_↑": chance,
        "oracle_accuracy_margin_over_chance_↑": oracle_accuracy - chance,
        "oracle_no_evidence_accuracy_↑": no_evidence_accuracy,
        "mixture_gold_log_score_gain_over_question_only_↑": mixture_audio_gain,
        "audio_dependent_records_↑": audio_dependent,
        "option_order_controls": option_values,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, selection = select_subset(args.source_manifest)
    root = args.results_root.resolve()
    subset = write_subset(root, rows)
    selection_path = root / "identifier_only_selection_receipt.json"
    if selection_path.is_file() and _read_json(selection_path) != selection:
        raise RuntimeError("existing Phi selection receipt changed")
    _atomic_json(selection_path, selection)
    output = root / "calibration"
    command = build_command(args, subset, output)
    validate_only = _run_validate_only(
        build_command(args, subset, output, validate_only=True)
    )
    prerequisite_state: dict[str, Any]
    try:
        forward = validate_forward_receipt(args.forward_receipt)
        prerequisite_state = {
            "passed": True,
            "receipt": _identity(args.forward_receipt),
            "forward_permission": forward["permission"],
        }
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        prerequisite_state = {"passed": False, "error": str(error)}
    dry_run = {
        "format": FORMAT,
        "mode": "execute" if args.execute else "dry_run",
        "selection": selection,
        "subset_manifest": _identity(subset),
        "validate_only": validate_only,
        "forward_prerequisite": prerequisite_state,
        "command": command,
        "predeclared_decision": {
            "pilot_go": (
                "all three oracle gold-log effects >0 ↑; oracle answerable accuracy "
                "≥ candidate-aware chance+0.05 ↑; oracle no-evidence accuracy ≥0.50 ↑; "
                "mixture gold-log gain over question-only >0 ↑; audio-dependent n≥3 ↑; "
                "all option-position integrity/invariance/gap gates pass"
            ),
            "confirmatory_pass_criteria": (
                "pilot_go plus the scene-family bootstrap 95% CI lower bound >0 ↑ "
                "for all three oracle gold-log contrasts"
            ),
            "pilot_go_without_confirmatory_ci": "expand oracle calibration",
            "on_confirmatory_pass": "authorize predicted-stem development pilot",
            "pilot_fail": "reject Phi as a QCES auditor on this pilot",
        },
    }
    if not args.execute:
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return
    if not prerequisite_state["passed"]:
        raise RuntimeError(
            "Phi calibration execution is blocked: " + prerequisite_state["error"]
        )
    state = gpu_state()
    require_idle_gpu(
        state, args.minimum_free_gpu_mib, args.maximum_gpu_utilization_percent
    )
    log_prefix = root / "logs/calibration"
    if (output / "evaluation_report.json").is_file():
        monitor = load_monitor(log_prefix, command)
    else:
        monitor = run_monitored(command, log_prefix, args.timeout_seconds)
    report_path = output / "evaluation_report.json"
    _, report = validate_calibration_output(args, subset, output)
    analysis = analyze_report(report)
    receipt = {
        **dry_run,
        "format": FORMAT,
        "mode": "completed",
        "gpu_preflight": state,
        "monitor": monitor,
        "evaluation_report": _identity(report_path),
        "run_metadata": _identity(output / "run_metadata.json"),
        "items": _identity(output / "items.jsonl"),
        "analysis": analysis,
        "permission": analysis["decision"],
        "paper_test_result": False,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    receipt_path = root / "phi4mm_oracle_calibration_receipt.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"wrote {receipt_path}")


if __name__ == "__main__":
    main()
