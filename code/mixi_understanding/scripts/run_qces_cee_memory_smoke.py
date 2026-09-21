#!/usr/bin/env python3
"""Run the predeclared QCES CEE AMP batch-3 CUDA memory smoke.

This is a ten-update resource/wiring audit on the training split, not a model
quality result.  The selected seed exercises surface, family, and question CEE
groups within those ten packed batches.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
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

from mixi_understanding.qces.counterfactual import (  # noqa: E402
    CounterfactualBatchSampler,
    build_counterfactual_group_plan,
)
from mixi_understanding.qces.data import QCESManifestDataset  # noqa: E402

FORMAT = "qces_cee_amp_batch3_memory_smoke_v1"
REQUIRED_GROUPS = ("surface", "family", "question")
CEE_WEIGHT_FLAGS = (
    "--surface-semantic-invariance-weight",
    "--surface-role-invariance-weight",
    "--surface-no-evidence-invariance-weight",
    "--surface-evidence-invariance-weight",
    "--family-temporal-delta-weight",
    "--family-evidence-delta-weight",
    "--family-no-evidence-transition-weight",
    "--question-temporal-delta-weight",
    "--question-evidence-delta-weight",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_train.jsonl",
    )
    parser.add_argument(
        "--foundation-cache",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_paper_caches/foundation_train",
    )
    parser.add_argument(
        "--semantic-targets",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_paper_caches/semantic_union_train.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_cee_memory_smoke_seed10/amp_batch3",
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
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=9_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps <= 0 or args.minimum_free_gpu_mib <= 0:
        parser.error("max steps and minimum free GPU memory must be positive")
    if not 0 <= args.maximum_gpu_utilization_percent <= 100:
        parser.error("maximum GPU utilization must lie in [0, 100]")
    return args


def gpu_state() -> dict[str, int]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        free, utilization, temperature = [
            int(item.strip())
            for item in completed.stdout.strip().splitlines()[0].split(",")
        ]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot query the CUDA device: {error}") from error
    return {
        "free_memory_mib_↑": free,
        "utilization_percent_↓": utilization,
        "temperature_celsius_↓": temperature,
    }


def require_idle_gpu(
    state: Mapping[str, int], minimum_free_mib: int, maximum_utilization: int
) -> None:
    if state["free_memory_mib_↑"] < minimum_free_mib:
        raise RuntimeError(
            f"insufficient free GPU memory: {state['free_memory_mib_↑']} MiB "
            f"< {minimum_free_mib} MiB"
        )
    if state["utilization_percent_↓"] > maximum_utilization:
        raise RuntimeError(
            "GPU is actively used by another workload: "
            f"{state['utilization_percent_↓']}% > {maximum_utilization}%"
        )


def require_inputs(args: argparse.Namespace) -> None:
    inputs = {
        "manifest": args.manifest,
        "foundation cache": args.foundation_cache,
        "semantic targets": args.semantic_targets,
        "AudioSep root": args.audiosep_root,
        "AudioSep config": args.audiosep_config,
        "AudioSep checkpoint": args.audiosep_checkpoint,
    }
    missing = [
        f"{name}: {path.resolve()}"
        for name, path in inputs.items()
        if not path.exists()
    ]
    if missing:
        raise RuntimeError("missing runner input(s): " + "; ".join(missing))


def audit_first_batches(manifest: Path, *, seed: int, max_steps: int) -> dict[str, Any]:
    records = QCESManifestDataset(
        manifest.resolve(), crop_samples=None, seed=seed
    ).records
    plan = build_counterfactual_group_plan(
        records, enable_surface=True, enable_family=True, enable_question=True
    )
    sampler = CounterfactualBatchSampler(plan, batch_size=3, seed=seed, shuffle=True)
    observed = {name: 0 for name in REQUIRED_GROUPS}
    schedule = []
    for step, indices in enumerate(list(sampler)[:max_steps], start=1):
        sample_ids = [plan.sample_ids[index] for index in indices]
        counts = plan.batch_groups(sample_ids)["counts"]
        for name in REQUIRED_GROUPS:
            observed[name] += int(counts[name])
        schedule.append(
            {
                "optimizer_step": step,
                "batch_records_↑": len(indices),
                "complete_group_counts_↑": counts,
            }
        )
    missing = [name for name, count in observed.items() if count <= 0]
    if missing:
        raise RuntimeError(
            f"first {max_steps} batches do not exercise CEE groups: {missing}"
        )
    return {
        "batch_size": 3,
        "optimizer_steps": len(schedule),
        "observed_complete_groups_↑": observed,
        "schedule": schedule,
        "full_group_plan_↑": {
            "surface_pairs": len(plan.surface_pairs),
            "primary_triplets": len(plan.primary_triplets),
            "question_pairs": len(plan.question_pairs),
            "records": len(plan.sample_ids),
        },
        "duplicate_record_calls_within_epoch_↓": 0,
    }


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(CODE_ROOT / "mixi_understanding/scripts/train_qces.py"),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
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
        str(args.foundation_cache.resolve()),
        "--foundation-semantic-mixing-mode",
        "question_residual",
        "--temporal-role-mode",
        "independent_sigmoid",
        "--semantic-separation-mode",
        "union_single",
        "--semantic-targets",
        str(args.semantic_targets.resolve()),
        "--semantic-weight",
        "2.0",
        "--weakest-role-weight",
        "0.25",
        "--role-relative-weight",
        "0.25",
        "--epochs",
        "1",
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
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--precision",
        "amp_fp16",
        "--selection-metric",
        "evidence_sd_sdr",
        "--no-separator-aware-refiner",
        "--deterministic",
    ]
    for flag in CEE_WEIGHT_FLAGS:
        command.extend((flag, "0.1"))
    return command


def validate_summary(summary: Mapping[str, Any], *, max_steps: int) -> dict[str, Any]:
    errors = []
    training = summary.get("training_config", {})
    cee_train = summary.get("counterfactual_evidence_equivariance", {}).get("train", {})
    precision = summary.get("precision", {})
    resources = summary.get("run_resources", {})
    audiosep = summary.get("audiosep", {})
    expected = (
        (summary.get("global_step") == max_steps, "wrong global_step"),
        (summary.get("stop_reason") == "max_steps", "wrong stop reason"),
        (training.get("batch_size") == 3, "batch size is not 3"),
        (training.get("precision") == "amp_fp16", "precision is not AMP"),
        (
            training.get("temporal_role_mode") == "independent_sigmoid",
            "temporal role mode is not overlap-aware",
        ),
        (bool(training.get("counterfactual_enabled")), "CEE is disabled"),
        (bool(cee_train.get("surface_invariance")), "surface CEE is disabled"),
        (bool(cee_train.get("family_equivariance")), "family CEE is disabled"),
        (bool(cee_train.get("question_equivariance")), "question CEE is disabled"),
        (precision.get("mode") == "amp_fp16", "AMP metadata is missing"),
        (bool(precision.get("autocast", {}).get("enabled")), "autocast is disabled"),
        (
            not bool(precision.get("audiosep_parameter_dtype_mutated")),
            "AudioSep dtype mutated",
        ),
        (
            audiosep.get("backbone_trainable_parameter_count") == 0,
            "AudioSep has trainable parameters",
        ),
    )
    errors.extend(message for passed, message in expected if not passed)
    allocated = resources.get("cuda_peak_allocated_bytes_down")
    reserved = resources.get("cuda_peak_reserved_bytes_down")
    for name, value in (("allocated", allocated), ("reserved", reserved)):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            errors.append(f"invalid CUDA peak {name} memory")
    return {
        "all_gates_passed": not errors,
        "errors": errors,
        "optimizer_steps_↑": summary.get("global_step"),
        "batch_size_↑": training.get("batch_size"),
        "cuda_peak_allocated_bytes_↓": allocated,
        "cuda_peak_reserved_bytes_↓": reserved,
        "frozen_audiosep_trainable_parameters_↓": audiosep.get(
            "backbone_trainable_parameter_count"
        ),
        "audiosep_dtype_mutations_↓": int(
            bool(precision.get("audiosep_parameter_dtype_mutated"))
        ),
        "amp_current_scale_↑": precision.get("gradient_scaler", {}).get(
            "current_scale"
        ),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require_inputs(args)
    schedule_audit = audit_first_batches(
        args.manifest, seed=args.seed, max_steps=args.max_steps
    )
    command = build_command(args)
    plan = {
        "format": FORMAT,
        "purpose": "resource_and_CEE_wiring_smoke_not_model_quality",
        "paper_result_eligible": False,
        "validation_or_test_records_accessed_↓": 0,
        "command": command,
        "schedule_audit": schedule_audit,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            "output-dir is non-empty; refuse to mix or overwrite experiment outputs"
        )
    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    print(json.dumps({"gpu_preflight": state}, sort_keys=True), flush=True)
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    summary_path = args.output_dir.resolve() / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = validate_summary(summary, max_steps=args.max_steps)
    receipt = {
        **plan,
        "gpu_preflight": state,
        "training_summary": str(summary_path),
        "gates": gates,
    }
    receipt_path = args.output_dir.resolve() / "cee_memory_smoke_receipt.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not gates["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
