#!/usr/bin/env python3
"""Run the contribution-identifying QCES ablation development screen.

This stage reuses the registered full/CEE-off runs from the schedule-matched
development pilot and trains only five additional one-seed variants.  It is a
fail-closed compute screen before the expensive three-seed full-data matrix;
it never accesses a test manifest and is not itself a paper result.
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
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.run_qces_cee_dev_pilot import (  # noqa: E402
    FORMAT as CEE_FORMAT,
    build_evaluate_command,
    build_train_command,
    candidate_execution_state,
    gpu_state,
    require_idle_gpu,
    select_listening_item_ids,
    validate_artifacts,
)


FORMAT = "qces_contribution_ablation_dev_screen_v1"
REFERENCE_VARIANTS = {
    "full_cee_on": "cee_on",
    "cee_off_matched": "cee_off_matched",
}
TRAINED_VARIANTS = (
    "direct_question_clap",
    "dual_role_factorization",
    "no_weakest_role_supervision",
    "no_explicit_no_evidence_supervision",
    "no_compactness_penalty",
)
VARIANTS = tuple(REFERENCE_VARIANTS) + TRAINED_VARIANTS
DUAL_CACHE_SHA256 = {
    "semantic_dual_train.pt": (
        "636e7439875e0b027161ba4f0b9033939070ffdcbb40dbb96738d22866052259"
    ),
    "semantic_dual_val.pt": (
        "f68212299f1e702d95d7cfacf2360b1ed035fb057e6d9d5c914c2c667d1e5a3d"
    ),
}
CORE_METRICS = {
    "evidence_sd_sdri_answerable": "maximize",
    "weakest_role_sd_sdr_answerable": "maximize",
    "answerable_temporal_iou": "maximize",
    "mean_no_evidence_retained_ratio": "minimize",
    "maximum_mixture_consistency_l1": "minimize",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python3.10"),
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
        "--cee-results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_cee_devpilot_seed2026",
    )
    parser.add_argument(
        "--cee-comparison",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_cee_devpilot_seed2026/comparison.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_ablation_dev_screen_seed2026",
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
    parser.add_argument("--listening-cases", type=int, default=0)
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


def cee_namespace(args: argparse.Namespace) -> SimpleNamespace:
    """Expose exactly the arguments consumed by the registered CEE builder."""

    return SimpleNamespace(
        python=args.python,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        cache_root=args.cache_root,
        results_root=args.cee_results_root,
        audiosep_root=args.audiosep_root,
        audiosep_config=args.audiosep_config,
        audiosep_checkpoint=args.audiosep_checkpoint,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        listening_cases=args.listening_cases,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        maximum_gpu_utilization_percent=args.maximum_gpu_utilization_percent,
        microfit_checkpoint=Path("unused"),
        microfit_health_receipt=Path("unused"),
        cee_memory_receipt=Path("unused"),
        execute=False,
    )


def _option_index(command: Sequence[str], flag: str) -> int | None:
    matches = [index for index, value in enumerate(command) if value == flag]
    if len(matches) > 1:
        raise ValueError(f"duplicate command option: {flag}")
    return matches[0] if matches else None


def set_option(command: list[str], flag: str, value: str) -> None:
    index = _option_index(command, flag)
    if index is None:
        command.extend((flag, value))
        return
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"option lacks a value: {flag}")
    command[index + 1] = value


def remove_option(command: list[str], flag: str) -> None:
    index = _option_index(command, flag)
    if index is None:
        return
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"option lacks a removable value: {flag}")
    del command[index : index + 2]


def add_boolean_option(command: list[str], flag: str) -> None:
    if _option_index(command, flag) is None:
        command.append(flag)


def remove_boolean_option(command: list[str], flag: str) -> None:
    index = _option_index(command, flag)
    if index is not None:
        del command[index]


def option_value(command: Sequence[str], flag: str) -> str | None:
    index = _option_index(command, flag)
    if index is None:
        return None
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        return None
    return command[index + 1]


def build_variant_train_command(args: argparse.Namespace, variant: str) -> list[str]:
    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"variant is not trainable here: {variant}")
    command = build_train_command(cee_namespace(args), "cee_on")
    set_option(
        command,
        "--output-dir",
        str((args.results_root.resolve() / f"{variant}_train")),
    )
    if variant == "direct_question_clap":
        remove_option(command, "--semantic-targets")
        remove_option(command, "--val-semantic-targets")
        set_option(command, "--semantic-weight", "0")
        add_boolean_option(command, "--freeze-semantic-adapter")
    elif variant == "dual_role_factorization":
        set_option(command, "--semantic-separation-mode", "dual_role")
        remove_option(command, "--semantic-targets")
        remove_option(command, "--val-semantic-targets")
        set_option(command, "--semantic-weight", "0")
        set_option(
            command,
            "--role-semantic-targets",
            str((args.cache_root / "semantic_dual_train.pt").resolve()),
        )
        set_option(
            command,
            "--val-role-semantic-targets",
            str((args.cache_root / "semantic_dual_val.pt").resolve()),
        )
        # Two role losses at 1.0 preserve the full candidate's total semantic
        # coefficient of 2.0; 0.1 trains only the required symmetric router.
        set_option(command, "--role-semantic-weight", "1.0")
        set_option(command, "--same-semantic-weight", "0.1")
    elif variant == "no_weakest_role_supervision":
        set_option(command, "--weakest-role-weight", "0")
    elif variant == "no_explicit_no_evidence_supervision":
        set_option(command, "--no-evidence-weight", "0")
        set_option(command, "--surface-no-evidence-invariance-weight", "0")
        set_option(command, "--family-no-evidence-transition-weight", "0")
    elif variant == "no_compactness_penalty":
        set_option(command, "--minimality-weight", "0")
    return command


def build_variant_evaluate_command(
    args: argparse.Namespace, variant: str, listening_ids: Sequence[str]
) -> list[str]:
    if variant not in TRAINED_VARIANTS:
        raise ValueError(f"variant is not evaluated here: {variant}")
    command = build_evaluate_command(cee_namespace(args), "cee_on", listening_ids)
    set_option(
        command,
        "--checkpoint",
        str(args.results_root.resolve() / f"{variant}_train/checkpoint.pt"),
    )
    set_option(
        command,
        "--output-dir",
        str(args.results_root.resolve() / f"{variant}_eval"),
    )
    return command


def build_dual_memory_smoke_command(args: argparse.Namespace) -> list[str]:
    """Use one complete AMP batch before committing to 512 dual-role updates."""

    command = build_variant_train_command(args, "dual_role_factorization")
    set_option(
        command,
        "--output-dir",
        str(args.results_root.resolve() / "dual_role_batch3_memory_smoke"),
    )
    for flag in (
        "--val-manifest",
        "--val-foundation-feature-cache",
        "--val-role-semantic-targets",
    ):
        remove_option(command, flag)
    set_option(command, "--epochs", "1")
    set_option(command, "--max-steps", "1")
    remove_boolean_option(command, "--save-every-epoch")
    return command


def promotion_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve().read_text("utf-8"))
        decision = payload.get("decision") if isinstance(payload, Mapping) else None
        integrity = payload.get("integrity") if isinstance(payload, Mapping) else None
        passed = bool(
            isinstance(payload, Mapping)
            and payload.get("format") == CEE_FORMAT
            and isinstance(decision, Mapping)
            and decision.get("promote_cee_to_full_seeded_run") is True
            and isinstance(integrity, Mapping)
            and integrity.get("all_passed") is True
        )
        return {
            "passed": passed,
            "receipt": _identity(path),
            "decision": dict(decision) if isinstance(decision, Mapping) else None,
            "error": None if passed else "CEE comparison has not authorized promotion",
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def validate_screen_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    base = validate_artifacts(cee_namespace(args))
    dual = {}
    for filename, expected in DUAL_CACHE_SHA256.items():
        path = args.cache_root.resolve() / filename
        if not path.is_file():
            raise RuntimeError(f"missing dual-role semantic cache: {path}")
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise RuntimeError(f"dual-role semantic cache hash mismatch: {path}")
        dual[filename] = identity
    sources = {
        "train_qces": _identity(CODE_ROOT / "mixi_understanding/scripts/train_qces.py"),
        "evaluate_qces": _identity(
            CODE_ROOT / "mixi_understanding/scripts/evaluate_qces.py"
        ),
    }
    return {"base": base, "dual_role_caches": dual, "source_code": sources}


def variant_paths(args: argparse.Namespace, variant: str) -> tuple[Path, Path]:
    reference = REFERENCE_VARIANTS.get(variant)
    if reference is not None:
        root = args.cee_results_root.resolve()
        return root / f"{reference}_train", root / f"{reference}_eval"
    root = args.results_root.resolve()
    return root / f"{variant}_train", root / f"{variant}_eval"


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def validate_dual_memory_smoke(summary: Mapping[str, Any]) -> dict[str, Any]:
    training = summary.get("training_config", {})
    resources = summary.get("run_resources", {})
    audiosep = summary.get("audiosep", {})
    checks = {
        "one_optimizer_step_↑": summary.get("global_step") == 1,
        "max_steps_stop_↑": summary.get("stop_reason") == "max_steps",
        "dual_role_mode_↑": training.get("semantic_separation_mode") == "dual_role",
        "amp_batch3_↑": (
            training.get("precision") == "amp_fp16" and training.get("batch_size") == 3
        ),
        "frozen_audiosep_↑": audiosep.get("backbone_trainable_parameter_count") == 0,
        "positive_peak_allocated_memory_↑": isinstance(
            resources.get("cuda_peak_allocated_bytes_down"), (int, float)
        )
        and resources.get("cuda_peak_allocated_bytes_down", 0) > 0,
        "positive_peak_reserved_memory_↑": isinstance(
            resources.get("cuda_peak_reserved_bytes_down"), (int, float)
        )
        and resources.get("cuda_peak_reserved_bytes_down", 0) > 0,
    }
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
        "cuda_peak_allocated_bytes_↓": resources.get("cuda_peak_allocated_bytes_down"),
        "cuda_peak_reserved_bytes_↓": resources.get("cuda_peak_reserved_bytes_down"),
    }


def validate_variant_contract(
    variant: str, summary: Mapping[str, Any], full: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    training = summary.get("training_config", {})
    losses = summary.get("loss_weights", {})
    full_losses = full.get("loss_weights", {})
    full_training = full.get("training_config", {})
    for name in (
        "backend",
        "epochs",
        "batch_size",
        "learning_rate",
        "dropout",
        "crop_seconds",
        "crop_samples",
        "max_steps",
        "num_workers",
        "seed",
        "precision",
        "deterministic",
        "temporal_role_mode",
        "foundation_feature_mode",
        "foundation_semantic_mixing_mode",
        "selection_metric",
        "selection_direction",
        "save_every_epoch",
        "separator_aware_refiner",
    ):
        if training.get(name) != full_training.get(name):
            errors.append(f"{variant}: training schedule differs: {name}")
    if training.get("precision") != "amp_fp16" or training.get("batch_size") != 3:
        errors.append(f"{variant}: registered reference is not AMP batch-3")
    if summary.get("manifest_sha256") != full.get("manifest_sha256"):
        errors.append(f"{variant}: train manifest differs")
    if summary.get("val_manifest_sha256") != full.get("val_manifest_sha256"):
        errors.append(f"{variant}: validation manifest differs")
    audiosep = summary.get("audiosep", {})
    full_audiosep = full.get("audiosep", {})
    if audiosep.get("checkpoint") != full_audiosep.get("checkpoint"):
        errors.append(f"{variant}: AudioSep checkpoint identity differs")
    initialization = summary.get("from_scratch_composer_initialization", {})
    full_initialization = full.get("from_scratch_composer_initialization", {})
    if initialization.get("aggregates", {}).get(
        "common_parameters_sha256"
    ) != full_initialization.get("aggregates", {}).get("common_parameters_sha256"):
        errors.append(f"{variant}: common composer initialization differs")

    if variant == "direct_question_clap":
        adapter = summary.get("foundation_features", {}).get("semantic_adapter", {})
        post = summary.get("foundation_features", {}).get(
            "semantic_adapter_post_training", {}
        )
        for phase, receipt in (("initial", adapter), ("post", post)):
            if (
                receipt.get("ablation") != "direct_full_question_clap"
                or receipt.get("audiosep_condition") != "normalize(full_question_clap)"
                or receipt.get("trainable_parameter_count") != 0
                or receipt.get("final_layer_nonzero_parameter_count") != 0
            ):
                errors.append(f"{variant}: invalid {phase} adapter receipt")
        if losses.get("semantic_alignment") != 0:
            errors.append(f"{variant}: semantic alignment is not zero")
    elif variant == "dual_role_factorization":
        if training.get("semantic_separation_mode") != "dual_role":
            errors.append(f"{variant}: semantic mode is not dual_role")
        expected = {
            "semantic_alignment": 0,
            "anchor_semantic_alignment": 1.0,
            "answer_semantic_alignment": 1.0,
            "same_semantic_classification": 0.1,
        }
        for name, value in expected.items():
            if losses.get(name) != value:
                errors.append(f"{variant}: wrong {name}")
    elif variant == "no_weakest_role_supervision":
        if losses.get("weakest_role_waveform") != 0:
            errors.append(f"{variant}: weakest-role weight is not zero")
    elif variant == "no_explicit_no_evidence_supervision":
        expected = {
            "no_evidence": 0,
            "surface_no_evidence_invariance": 0,
            "family_no_evidence_transition": 0,
        }
        for name, value in expected.items():
            if losses.get(name) != value:
                errors.append(f"{variant}: wrong {name}")
    elif variant == "no_compactness_penalty":
        if losses.get("minimality") != 0:
            errors.append(f"{variant}: minimality weight is not zero")
    elif variant == "cee_off_matched":
        cee = summary.get("counterfactual_evidence_equivariance", {})
        if (
            cee.get("paired_objective_enabled") is not False
            or cee.get("forced_schedule_matched_control") is not True
        ):
            errors.append(f"{variant}: CEE-off schedule contract failed")
    elif variant == "full_cee_on":
        cee = summary.get("counterfactual_evidence_equivariance", {})
        if cee.get("paired_objective_enabled") is not True:
            errors.append(f"{variant}: CEE objective is disabled")

    if variant not in {"direct_question_clap", "dual_role_factorization"}:
        for name, value in full_losses.items():
            allowed = {
                "no_weakest_role_supervision": {"weakest_role_waveform"},
                "no_explicit_no_evidence_supervision": {
                    "no_evidence",
                    "surface_no_evidence_invariance",
                    "family_no_evidence_transition",
                },
                "no_compactness_penalty": {"minimality"},
                "cee_off_matched": {
                    name
                    for name in full_losses
                    if name.startswith(("surface_", "family_", "question_"))
                },
                "full_cee_on": set(),
            }.get(variant, set())
            if name not in allowed and losses.get(name) != value:
                errors.append(f"{variant}: unrelated loss changed: {name}")
    return errors


def compare_variants(args: argparse.Namespace) -> dict[str, Any]:
    summaries = {}
    evaluations = {}
    for variant in VARIANTS:
        train_dir, eval_dir = variant_paths(args, variant)
        summaries[variant] = _read_json(train_dir / "summary.json")
        evaluations[variant] = _read_json(eval_dir / "evaluation_report.json")
    full = summaries["full_cee_on"]
    errors = []
    for variant, summary in summaries.items():
        errors.extend(validate_variant_contract(variant, summary, full))

    metrics: dict[str, Any] = {}
    full_eval = evaluations["full_cee_on"].get("summary", {})
    for metric, direction in CORE_METRICS.items():
        arrow = "↑" if direction == "maximize" else "↓"
        values = {
            variant: float(evaluations[variant]["summary"][metric])
            for variant in VARIANTS
        }
        full_value = float(full_eval[metric])
        metrics[f"{metric}_{arrow}"] = {
            **values,
            "direction": direction,
            "full_minus_variant_effect_oriented_↑": {
                variant: (
                    full_value - value
                    if direction == "maximize"
                    else value - full_value
                )
                for variant, value in values.items()
                if variant != "full_cee_on"
            },
        }
    anti_collapse = {
        variant: {
            "evidence_sd_sdri_positive_↑": float(
                evaluations[variant]["summary"]["evidence_sd_sdri_answerable"]
            )
            > 0,
            "temporal_iou_at_least_0.27_↑": float(
                evaluations[variant]["summary"]["answerable_temporal_iou"]
            )
            >= 0.27,
            "no_evidence_retention_at_most_0.10_↓": float(
                evaluations[variant]["summary"]["mean_no_evidence_retained_ratio"]
            )
            <= 0.10,
            "mixture_error_at_most_1e-5_↓": float(
                evaluations[variant]["summary"]["maximum_mixture_consistency_l1"]
            )
            <= 1e-5,
        }
        for variant in VARIANTS
    }
    return {
        "format": FORMAT,
        "scope": "heldout_development_ablation_screen_not_test_result",
        "integrity": {"all_passed": not errors, "errors": errors},
        "metrics": metrics,
        "anti_collapse_diagnostics": anti_collapse,
        "decision": {
            "screen_complete_↑": not errors,
            "three_seed_registry_may_be_frozen_↑": not errors,
            "ablation_failure_is_expected_evidence_not_a_screen_failure": True,
            "test_access_authorized": False,
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }


def _run(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=environment, check=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = validate_screen_artifacts(args)
    promotion = promotion_state(args.cee_comparison)
    listening_ids = select_listening_item_ids(args.val_manifest, args.listening_cases)
    commands = {
        variant: {
            "train": build_variant_train_command(args, variant),
            "evaluate": build_variant_evaluate_command(args, variant, listening_ids),
        }
        for variant in TRAINED_VARIANTS
    }
    dual_memory_command = build_dual_memory_smoke_command(args)
    plan = {
        "format": FORMAT,
        "purpose": "one_seed_contribution_screen_before_three_seed_full_matrix",
        "paper_result_eligible": False,
        "test_records_accessed_↓": 0,
        "reference_variants": REFERENCE_VARIANTS,
        "trained_variants": list(TRAINED_VARIANTS),
        "promotion_prerequisite": promotion,
        "artifacts": artifacts,
        "commands": commands,
        "dual_role_batch3_memory_smoke_command": dual_memory_command,
        "semantic_weight_matching": {
            "union_single_total_coefficient": 2.0,
            "dual_role_anchor_coefficient": 1.0,
            "dual_role_answer_coefficient": 1.0,
            "dual_role_same_semantic_router_coefficient": 0.1,
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if not promotion["passed"]:
        raise SystemExit("CEE promotion prerequisite failed: " + promotion["error"])
    plan_path = args.results_root.resolve() / "frozen_ablation_screen_plan.json"
    if plan_path.is_file():
        if _read_json(plan_path) != plan:
            raise SystemExit(f"existing frozen ablation plan changed: {plan_path}")
    else:
        _atomic_json(plan_path, plan)
    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    smoke_dir = args.results_root.resolve() / "dual_role_batch3_memory_smoke"
    smoke_receipt_path = smoke_dir / "dual_role_memory_receipt.json"
    if smoke_receipt_path.is_file():
        smoke_receipt = _read_json(smoke_receipt_path)
        if (
            smoke_receipt.get("format") != FORMAT
            or smoke_receipt.get("gates", {}).get("all_passed") is not True
        ):
            raise SystemExit(f"invalid dual-role memory receipt: {smoke_receipt_path}")
    else:
        if smoke_dir.is_dir() and any(smoke_dir.iterdir()):
            raise SystemExit(
                f"partial dual-role memory smoke requires audit: {smoke_dir}"
            )
        _run(dual_memory_command)
        summary_path = smoke_dir / "summary.json"
        smoke_gates = validate_dual_memory_smoke(_read_json(summary_path))
        smoke_receipt = {
            "format": FORMAT,
            "purpose": "dual_role_amp_batch3_resource_gate_not_quality_result",
            "command": dual_memory_command,
            "training_summary": _identity(summary_path),
            "gates": smoke_gates,
            "metric_direction_legend": {
                "↑": "higher is better",
                "↓": "lower is better",
            },
        }
        _atomic_json(smoke_receipt_path, smoke_receipt)
        if not smoke_gates["all_passed"]:
            raise SystemExit(2)
    for variant in TRAINED_VARIANTS:
        try:
            stage = candidate_execution_state(args.results_root.resolve(), variant)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not stage["train_complete"]:
            _run(commands[variant]["train"])
        if not stage["evaluation_complete"]:
            _run(commands[variant]["evaluate"])
    comparison = compare_variants(args)
    comparison["gpu_preflight"] = state
    comparison["dual_role_memory_smoke"] = _identity(smoke_receipt_path)
    comparison["frozen_plan"] = _identity(plan_path)
    comparison_path = args.results_root.resolve() / "ablation_screen_comparison.json"
    _atomic_json(comparison_path, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    if not comparison["integrity"]["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
