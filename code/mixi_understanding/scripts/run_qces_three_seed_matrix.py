#!/usr/bin/env python3
"""Freeze and execute the full-data three-seed QCES ablation matrix.

The registry contains the promoted full system plus six contribution-identifying
variants for seeds 2026/2027/2028.  It uses train and held-out validation only;
test manifests are structurally prohibited.  Execution is seed-resumable and
requires the integrity-clean one-seed ablation development screen.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.run_qces_ablation_dev_screen import (  # noqa: E402
    CORE_METRICS,
    FORMAT as SCREEN_FORMAT,
    TRAINED_VARIANTS,
    VARIANTS,
    build_variant_evaluate_command,
    build_variant_train_command,
    cee_namespace,
    set_option,
    validate_variant_contract,
)
from mixi_understanding.scripts.run_qces_cee_dev_pilot import (  # noqa: E402
    build_evaluate_command,
    build_train_command,
    candidate_execution_state,
    gpu_state,
    require_idle_gpu,
)


FORMAT = "qces_full_validation_three_seed_matrix_v1"
SEEDS = (2026, 2027, 2028)
FULL_ARTIFACT_SHA256 = {
    "train_manifest": "08856a7ecd65234a0b8a9d41d1cda4f72738bf5f8e51a777b56bdc2abebd4cfa",
    "val_manifest": "6bd26712b7afe069bf0797bec697380344589d6ef7a906077c34ff21f63a183a",
    "dataset_config": "f67d6001654d34796b36514cd46727a388fb60483a0b48bacb11d7a5c72a1e7f",
    "validation_report": "1071c51529472aa532d374da9045fb6ccbcd269651ac3e5649eae91cdd8874dd",
    "foundation_train_receipt": "bf4f4373a45cb20af86f11dae3b43855cfbf5cdb9f67f9b6694023f4675f71fc",
    "foundation_train_question": "fb739f5ba3005c7408bff36171fd540c3dac46d835f86326836f2961f756e4e7",
    "foundation_train_scene": "7460aa00c2b826ff22ffe26d54070812377242098a945233a09638f98abfb341",
    "foundation_val_receipt": "72ed5611a93484bfa3b858c68317fb7125cca0961e79ee0e4a220a00fe3e6fa6",
    "foundation_val_question": "eee947503b990f47ce25cde4442550648d70e59b38a43968d0a0a778943a1377",
    "foundation_val_scene": "da647a1b6147d585489c3a94cbdc0dc29b82de13470431862027ed7770c07d9d",
    "semantic_union_train": "e3d537e19ff447a1dccc74e2ea9b2f35a19b80ccaf7c5093a7c35171b2da1f68",
    "semantic_union_val": "08ccefedaf6e82638004bc0a5040494ff7009ca8583f00200483685843b27f80",
    "semantic_dual_train": "b9a764340bda32e027c6d9c35c58acbf9a116540a44ee3cac4de0148df3f5265",
    "semantic_dual_val": "be8ff6870f02d896541e10b33f760a1299bcd70c1866e83f23c2be247c08ff10",
    "audiosep_config": "e7e2e1a089d1de5b58ee0ddeae978f5c8a4649ae0ddea724301363a1427f7f52",
    "audiosep_checkpoint": "37f1691fb067e2575f1ad1cfbfe44b7b3da18e52f33fcb2b0937b72952f11ba1",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-registry", action="store_true")
    mode.add_argument("--execute-seed", type=int, choices=SEEDS)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python3.10"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_train.jsonl",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_val.jsonl",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_paper_caches",
    )
    parser.add_argument(
        "--screen-comparison",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_ablation_dev_screen_seed2026/"
        "ablation_screen_comparison.json",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_full_three_seed_matrix",
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="0 runs the complete registered grouped epoch (default)",
    )
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=12_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.max_steps < 0:
        parser.error("epochs must be positive and max steps non-negative")
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


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.resolve().read_text("utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "dataset_config": args.train_manifest.parent / "dataset_config.json",
        "validation_report": args.train_manifest.parent / "validation_report.json",
        "foundation_train_receipt": args.cache_root
        / "foundation_train/cache_receipt.json",
        "foundation_train_question": args.cache_root
        / "foundation_train/question_features.pt",
        "foundation_train_scene": args.cache_root
        / "foundation_train/scene_audio_features.pt",
        "foundation_val_receipt": args.cache_root / "foundation_val/cache_receipt.json",
        "foundation_val_question": args.cache_root
        / "foundation_val/question_features.pt",
        "foundation_val_scene": args.cache_root
        / "foundation_val/scene_audio_features.pt",
        "semantic_union_train": args.cache_root / "semantic_union_train.pt",
        "semantic_union_val": args.cache_root / "semantic_union_val.pt",
        "semantic_dual_train": args.cache_root / "semantic_dual_train.pt",
        "semantic_dual_val": args.cache_root / "semantic_dual_val.pt",
        "audiosep_config": args.audiosep_config,
        "audiosep_checkpoint": args.audiosep_checkpoint,
    }


def validate_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    identities = {}
    for name, path in artifact_paths(args).items():
        if not path.is_file():
            raise RuntimeError(f"missing full-matrix artifact: {name}: {path}")
        identity = _identity(path)
        if identity["sha256"] != FULL_ARTIFACT_SHA256[name]:
            raise RuntimeError(f"full-matrix artifact hash mismatch: {name}")
        identities[name] = identity
    if not args.python.is_file():
        raise RuntimeError(f"Python executable is missing: {args.python}")
    if not args.audiosep_root.is_dir():
        raise RuntimeError(f"AudioSep repository is missing: {args.audiosep_root}")
    report = _read_json(args.train_manifest.parent / "validation_report.json")
    required = {
        "profile_is_paper_↑": report.get("profile") == "paper",
        "synthetic_scale_gate_↑": report.get("synthetic_paper_scale_gate_passed")
        is True,
        "license_verified_↑": report.get("license_status") == "verified",
        "source_overlap_zero_↓": report.get("source_split_overlap_count_down") == 0,
        "scene_family_overlap_zero_↓": report.get(
            "scene_family_split_overlap_count_down"
        )
        == 0,
        "template_overlap_zero_↓": report.get("template_partition_overlap_count_down")
        == 0,
    }
    if not all(required.values()):
        raise RuntimeError(f"full benchmark integrity gate failed: {required}")
    return {
        "files": identities,
        "python": _identity(args.python),
        "benchmark_integrity": required,
        "source_code": {
            "train_qces": _identity(
                CODE_ROOT / "mixi_understanding/scripts/train_qces.py"
            ),
            "evaluate_qces": _identity(
                CODE_ROOT / "mixi_understanding/scripts/evaluate_qces.py"
            ),
        },
    }


def screen_prerequisite(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json(path)
        integrity = payload.get("integrity", {})
        decision = payload.get("decision", {})
        memory = payload.get("dual_role_memory_smoke")
        passed = bool(
            payload.get("format") == SCREEN_FORMAT
            and isinstance(integrity, Mapping)
            and integrity.get("all_passed") is True
            and isinstance(decision, Mapping)
            and decision.get("screen_complete_↑") is True
            and decision.get("three_seed_registry_may_be_frozen_↑") is True
            and isinstance(memory, Mapping)
        )
        return {
            "passed": passed,
            "receipt": _identity(path),
            "error": None if passed else "ablation screen has not authorized registry",
        }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def seed_namespace(args: argparse.Namespace, seed: int) -> SimpleNamespace:
    seed_root = args.results_root.resolve() / f"seed_{seed}"
    return SimpleNamespace(
        python=args.python,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        cache_root=args.cache_root,
        results_root=seed_root,
        cee_results_root=seed_root,
        audiosep_root=args.audiosep_root,
        audiosep_config=args.audiosep_config,
        audiosep_checkpoint=args.audiosep_checkpoint,
        seed=seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        listening_cases=0,
        minimum_free_gpu_mib=args.minimum_free_gpu_mib,
        maximum_gpu_utilization_percent=args.maximum_gpu_utilization_percent,
        execute=False,
        microfit_checkpoint=Path("unused"),
        microfit_health_receipt=Path("unused"),
        cee_memory_receipt=Path("unused"),
    )


def _bind_output_paths(
    command: list[str], seed_root: Path, variant: str, *, evaluate: bool
) -> list[str]:
    if evaluate:
        set_option(
            command,
            "--checkpoint",
            str(seed_root / f"{variant}_train/checkpoint.pt"),
        )
        set_option(command, "--output-dir", str(seed_root / f"{variant}_eval"))
    else:
        set_option(command, "--output-dir", str(seed_root / f"{variant}_train"))
    return command


def build_seed_commands(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    namespace = seed_namespace(args, seed)
    seed_root = namespace.results_root.resolve()
    commands: dict[str, Any] = {}
    for variant, cee_candidate in (
        ("full_cee_on", "cee_on"),
        ("cee_off_matched", "cee_off_matched"),
    ):
        train = build_train_command(cee_namespace(namespace), cee_candidate)
        evaluate = build_evaluate_command(cee_namespace(namespace), cee_candidate, ())
        commands[variant] = {
            "train": _bind_output_paths(train, seed_root, variant, evaluate=False),
            "evaluate": _bind_output_paths(evaluate, seed_root, variant, evaluate=True),
        }
    for variant in TRAINED_VARIANTS:
        train = build_variant_train_command(namespace, variant)
        evaluate = build_variant_evaluate_command(namespace, variant, ())
        commands[variant] = {
            "train": _bind_output_paths(train, seed_root, variant, evaluate=False),
            "evaluate": _bind_output_paths(evaluate, seed_root, variant, evaluate=True),
        }
    if set(commands) != set(VARIANTS):
        raise RuntimeError("three-seed registry does not cover the frozen variants")
    forbidden = (
        "qces_test_iid.jsonl",
        "qces_test_compositional_ood.jsonl",
        "qces_test_label_ood.jsonl",
    )
    flattened = json.dumps(commands, sort_keys=True)
    if any(name in flattened for name in forbidden):
        raise RuntimeError("test manifest leaked into the validation registry")
    return commands


def proposed_registry(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "scope": "full_train_and_heldout_validation_only",
        "paper_result_eligible": False,
        "test_records_accessed_↓": 0,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "training_budget": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "max_steps_zero_means_complete_grouped_epoch": args.max_steps == 0,
            "batch_size": 3,
            "precision": "amp_fp16",
        },
        "screen_prerequisite": screen_prerequisite(args.screen_comparison),
        "artifacts": validate_artifacts(args),
        "commands": {str(seed): build_seed_commands(args, seed) for seed in SEEDS},
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }


def _run(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(list(command), cwd=PROJECT_ROOT, env=environment, check=True)


def compare_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    root = args.results_root.resolve() / f"seed_{seed}"
    summaries = {
        variant: _read_json(root / f"{variant}_train/summary.json")
        for variant in VARIANTS
    }
    evaluations = {
        variant: _read_json(root / f"{variant}_eval/evaluation_report.json")
        for variant in VARIANTS
    }
    full = summaries["full_cee_on"]
    errors = []
    for variant, summary in summaries.items():
        errors.extend(validate_variant_contract(variant, summary, full))
        if summary.get("training_config", {}).get("seed") != seed:
            errors.append(f"{variant}: summary seed differs from registry seed {seed}")
    metrics = {}
    for metric, direction in CORE_METRICS.items():
        arrow = "↑" if direction == "maximize" else "↓"
        values = {
            variant: float(evaluations[variant]["summary"][metric])
            for variant in VARIANTS
        }
        full_value = values["full_cee_on"]
        metrics[f"{metric}_{arrow}"] = {
            "values": values,
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
    return {
        "format": FORMAT,
        "seed": seed,
        "scope": "full_train_heldout_validation_not_test",
        "integrity": {"all_passed": not errors, "errors": errors},
        "metrics": metrics,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }


def aggregate_seeds(args: argparse.Namespace) -> dict[str, Any] | None:
    paths = [
        args.results_root.resolve() / f"seed_{seed}/seed_comparison.json"
        for seed in SEEDS
    ]
    if not all(path.is_file() for path in paths):
        return None
    reports = [_read_json(path) for path in paths]
    errors = []
    for seed, report in zip(SEEDS, reports):
        if report.get("format") != FORMAT or report.get("seed") != seed:
            errors.append(f"invalid seed comparison: {seed}")
        if report.get("integrity", {}).get("all_passed") is not True:
            errors.append(f"seed integrity failed: {seed}")
    aggregates = {}
    for metric, direction in CORE_METRICS.items():
        arrow = "↑" if direction == "maximize" else "↓"
        display = f"{metric}_{arrow}"
        per_variant = {}
        for variant in VARIANTS:
            values = [
                float(report["metrics"][display]["values"][variant])
                for report in reports
            ]
            if not all(math.isfinite(value) for value in values):
                errors.append(f"non-finite aggregate: {display}/{variant}")
            per_variant[variant] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values),
                "values_by_seed": dict(zip(map(str, SEEDS), values)),
            }
        full_values = [
            float(report["metrics"][display]["values"]["full_cee_on"])
            for report in reports
        ]
        effects = {}
        for variant in VARIANTS:
            if variant == "full_cee_on":
                continue
            variant_values = [
                float(report["metrics"][display]["values"][variant])
                for report in reports
            ]
            oriented = [
                full - ablated if direction == "maximize" else ablated - full
                for full, ablated in zip(full_values, variant_values)
            ]
            effects[variant] = {
                "mean_full_minus_variant_effect_oriented_↑": statistics.fmean(oriented),
                "sample_std": statistics.stdev(oriented),
                "paired_values_by_seed": dict(zip(map(str, SEEDS), oriented)),
            }
        aggregates[display] = {
            "direction": direction,
            "variants": per_variant,
            "paired_full_minus_variant_effects_↑": effects,
        }
    return {
        "format": FORMAT,
        "scope": "three_seed_full_train_heldout_validation_not_locked_test",
        "integrity": {"all_passed": not errors, "errors": errors},
        "seeds": list(SEEDS),
        "aggregates": aggregates,
        "test_access_authorized": False,
        "next_gate": (
            "freeze one promoted checkpoint policy, complete non-oracle baselines "
            "and real/human audit before any one-time locked-test evaluation"
        ),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    registry = proposed_registry(args)
    print(json.dumps(registry, indent=2, sort_keys=True), flush=True)
    if not args.freeze_registry and args.execute_seed is None:
        return
    prerequisite = registry["screen_prerequisite"]
    if not prerequisite["passed"]:
        raise SystemExit(
            "three-seed registry prerequisite failed: " + prerequisite["error"]
        )
    registry_path = args.results_root.resolve() / "frozen_three_seed_registry.json"
    if registry_path.is_file():
        if _read_json(registry_path) != registry:
            raise SystemExit(f"frozen three-seed registry changed: {registry_path}")
    else:
        _atomic_json(registry_path, registry)
    if args.freeze_registry:
        print(f"froze {registry_path}")
        return

    assert args.execute_seed is not None
    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    seed = args.execute_seed
    seed_root = args.results_root.resolve() / f"seed_{seed}"
    commands = registry["commands"][str(seed)]
    for variant in VARIANTS:
        try:
            stage = candidate_execution_state(seed_root, variant)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        if not stage["train_complete"]:
            _run(commands[variant]["train"])
        if not stage["evaluation_complete"]:
            _run(commands[variant]["evaluate"])
    comparison = compare_seed(args, seed)
    comparison["gpu_preflight"] = state
    comparison["frozen_registry"] = _identity(registry_path)
    comparison_path = seed_root / "seed_comparison.json"
    _atomic_json(comparison_path, comparison)
    if not comparison["integrity"]["all_passed"]:
        raise SystemExit(2)
    aggregate = aggregate_seeds(args)
    if aggregate is not None:
        aggregate["frozen_registry"] = _identity(registry_path)
        aggregate_path = args.results_root.resolve() / "three_seed_summary.json"
        _atomic_json(aggregate_path, aggregate)
        print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
