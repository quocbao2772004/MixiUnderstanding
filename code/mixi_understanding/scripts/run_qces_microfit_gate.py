#!/usr/bin/env python3
"""Run the checkpointed QCES micro-overfit anti-collapse gate end to end.

This is an engineering sanity experiment, not a held-out paper result.  It
trains only on the frozen identifier-selected microfit manifest, evaluates the
same records, and authorizes the Streamlit demo only when the controller can
memorize temporal evidence *and* produce a non-collapsed waveform.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
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
    build_health_receipt,
)


FORMAT_VERSION = "qces_microfit_gate_runner_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_microfit_train_seed2028.jsonl",
    )
    parser.add_argument(
        "--foundation-cache",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_microfit_seed2028/foundation_train",
    )
    parser.add_argument(
        "--semantic-targets",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_microfit_seed2028/semantic_union_train.pt",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_microfit_seed2028/union_train_balanced_v3",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_v5_microfit_seed2028/union_eval_balanced_v3",
    )
    parser.add_argument(
        "--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep"
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
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1_536)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--semantic-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=7_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    parser.add_argument(
        "--listening-cases",
        type=int,
        default=4,
        help=(
            "Preselect this many manifest rows for WAV rendering during the same "
            "numeric evaluation; 0 keeps the numeric-only path."
        ),
    )
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="skip training and evaluate train-dir/checkpoint.pt",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.max_steps <= 0:
        parser.error("--epochs and --max-steps must be positive")
    if args.learning_rate <= 0 or args.semantic_weight <= 0:
        parser.error("learning rate and semantic weight must be positive")
    if args.minimum_free_gpu_mib <= 0:
        parser.error("minimum free GPU memory must be positive")
    if not 0 <= args.maximum_gpu_utilization_percent <= 100:
        parser.error("maximum GPU utilization must lie in [0, 100]")
    if args.listening_cases < 0:
        parser.error("--listening-cases must be non-negative")
    return args


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_health_outcome(
    train_dir: Path, eval_dir: Path, health: Mapping[str, Any]
) -> dict[str, str | None]:
    """Emit a demo receipt only for a passing checkpoint.

    Failed metrics are still persisted for diagnosis, but deliberately outside
    the checkpoint directory and under a filename the Streamlit app will never
    accept as authorization.
    """

    if bool(health.get("all_passed")):
        receipt_path = train_dir.resolve() / "demo_health_receipt.json"
        _atomic_json(receipt_path, health)
        return {
            "health_receipt": str(receipt_path),
            "failed_health_audit": None,
        }
    failed_path = eval_dir.resolve() / "failed_health_audit.json"
    _atomic_json(failed_path, health)
    return {
        "health_receipt": None,
        "failed_health_audit": str(failed_path),
    }


def analyze_microfit_result(
    training_summary: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Localize a failed memorization gate before choosing another experiment.

    The returned semantic values are diagnostics rather than extra deployment
    gates.  Only the fingerprint-bound health receipt authorizes the demo.
    """

    history = training_summary.get("history")
    epochs = history if isinstance(history, list) else []
    first = epochs[0] if epochs and isinstance(epochs[0], Mapping) else {}
    last = epochs[-1] if epochs and isinstance(epochs[-1], Mapping) else {}
    report_summary = evaluation_report.get("summary")
    evaluated = report_summary if isinstance(report_summary, Mapping) else {}

    def semantic_cosine(row: Mapping[str, Any]) -> float | None:
        loss = row.get("semantic_alignment")
        if isinstance(loss, (int, float)):
            return 1.0 - float(loss)
        return None

    first_cosine = semantic_cosine(first)
    final_cosine = semantic_cosine(last)
    cosine_gain = (
        final_cosine - first_cosine
        if first_cosine is not None and final_cosine is not None
        else None
    )
    failed = {
        str(gate.get("metric"))
        for gate in health.get("gates", [])
        if isinstance(gate, Mapping) and not bool(gate.get("passed"))
    }
    semantic_stalled = bool(
        final_cosine is not None
        and final_cosine < 0.70
        and (cosine_gain is None or cosine_gain < 0.02)
    )
    recommendations = []
    if "maximum_mixture_consistency_l1" in failed:
        recommendations.append(
            "stop experiments and audit the E/R arithmetic path before retraining"
        )
    if "answerable_temporal_iou" in failed:
        recommendations.append(
            (
                "audit question-residual semantic gradients/targets before changing "
                "the temporal controller"
                if semantic_stalled
                else "run the fixed-raw-stem overlap-aware temporal-refiner probe"
            )
        )
    if (
        "evidence_sd_sdri_answerable" in failed
        and "answerable_temporal_iou" not in failed
    ):
        recommendations.append(
            "factorize the frozen AudioSep raw stem from the learned temporal gate"
        )
    if failed & {"no_evidence_balanced_accuracy", "no_evidence_auroc"}:
        recommendations.append(
            "inspect train-balanced no-evidence logits and answerable false-silence errors"
        )
    if "mean_no_evidence_retained_ratio" in failed:
        recommendations.append(
            "inspect negative temporal-mask suppression; do not tune only the clip logit"
        )
    if not failed:
        recommendations.append(
            "authorize one Streamlit listening case, then run held-out validation"
        )
    return {
        "scope": "micro_overfit_failure_localization_not_heldout_evidence",
        "semantic_target_cosine_first_epoch_↑": first_cosine,
        "semantic_target_cosine_final_epoch_↑": final_cosine,
        "semantic_target_cosine_gain_↑": cosine_gain,
        "semantic_stalled_diagnostic": semantic_stalled,
        "answerable_temporal_iou_↑": evaluated.get("answerable_temporal_iou"),
        "evidence_sd_sdri_answerable_↑": evaluated.get("evidence_sd_sdri_answerable"),
        "no_evidence_balanced_accuracy_↑": evaluated.get(
            "no_evidence_balanced_accuracy"
        ),
        "no_evidence_auroc_↑": evaluated.get("no_evidence_auroc"),
        "mean_no_evidence_retained_ratio_↓": evaluated.get(
            "mean_no_evidence_retained_ratio"
        ),
        "maximum_mixture_consistency_l1_↓": evaluated.get(
            "maximum_mixture_consistency_l1"
        ),
        "failed_health_metrics": sorted(failed),
        "recommended_next_actions_in_order": recommendations,
        "decision_rule": (
            "arithmetic defect first; then semantic/temporal localization; then "
            "separator waveform factorization; negative suppression independently"
        ),
    }


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
        row = completed.stdout.strip().splitlines()[0]
        free, utilization, temperature = [int(item.strip()) for item in row.split(",")]
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
            "insufficient free GPU memory: "
            f"{state['free_memory_mib_↑']} MiB < {minimum_free_mib} MiB"
        )
    if state["utilization_percent_↓"] > maximum_utilization:
        raise RuntimeError(
            "GPU is actively used by another workload: "
            f"{state['utilization_percent_↓']}% > {maximum_utilization}%"
        )


def _require_inputs(args: argparse.Namespace) -> None:
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


def select_listening_item_ids(manifest: Path, count: int) -> list[str]:
    """Preselect a relation/status-stratified packet without reading results."""

    if count <= 0:
        return []
    rows: list[dict[str, Any]] = []
    with manifest.resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
                raise RuntimeError(
                    f"invalid listening-selection row at {manifest}:{line_number}"
                )
            rows.append(payload)
    rows.sort(key=lambda item: str(item["id"]))
    priorities = (
        ("after", False),
        ("before", False),
        ("first", False),
        ("first", True),
    )
    selected: list[str] = []
    for relation, no_evidence in priorities:
        match = next(
            (
                str(item["id"])
                for item in rows
                if item.get("relation") == relation
                and bool(item.get("no_evidence")) == no_evidence
                and str(item["id"]) not in selected
            ),
            None,
        )
        if match is not None:
            selected.append(match)
        if len(selected) >= count:
            return selected
    selected.extend(str(item["id"]) for item in rows if str(item["id"]) not in selected)
    if len(selected) < count:
        raise RuntimeError(
            f"requested {count} listening cases from only {len(selected)} records"
        )
    return selected[:count]


def build_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    checkpoint = args.train_dir.resolve() / "checkpoint.pt"
    train = [
        sys.executable,
        str(CODE_ROOT / "mixi_understanding/scripts/train_qces.py"),
        "--output-dir",
        str(args.train_dir.resolve()),
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
        "--weakest-role-weight",
        "0.25",
        "--role-relative-weight",
        "0.25",
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        "1",
        "--learning-rate",
        str(args.learning_rate),
        "--dropout",
        "0.1",
        "--crop-seconds",
        "10",
        "--selection-metric",
        "evidence_sd_sdr",
        "--early-stopping-patience",
        "0",
        "--selection-min-delta",
        "0",
        "--num-workers",
        "0",
        "--log-every-epochs",
        "1",
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--precision",
        "fp32",
        "--semantic-separation-mode",
        "union_single",
        "--semantic-targets",
        str(args.semantic_targets.resolve()),
        "--semantic-weight",
        str(args.semantic_weight),
        "--role-semantic-weight",
        "0",
        "--same-semantic-weight",
        "0",
        "--no-separator-aware-refiner",
        "--save-every-epoch",
        "--deterministic",
    ]
    listening_item_ids = select_listening_item_ids(args.manifest, args.listening_cases)
    evaluate = [
        sys.executable,
        str(CODE_ROOT / "mixi_understanding/scripts/evaluate_qces.py"),
        "--checkpoint",
        str(checkpoint),
        "--audiosep-root",
        str(args.audiosep_root.resolve()),
        "--audiosep-config",
        str(args.audiosep_config.resolve()),
        "--audiosep-checkpoint",
        str(args.audiosep_checkpoint.resolve()),
        "--foundation-feature-cache",
        str(args.foundation_cache.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output-dir",
        str(args.eval_dir.resolve()),
        "--batch-size",
        "1",
        "--device",
        "cuda",
    ]
    if listening_item_ids:
        for item_id in listening_item_ids:
            evaluate.extend(("--render-item-id", item_id))
    else:
        evaluate.append("--no-render-audio")
    return {"train": train, "evaluate": evaluate}


def _run(command: Sequence[str]) -> None:
    environment = os.environ.copy()
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _require_inputs(args)
    commands = build_commands(args)
    listening_item_ids = select_listening_item_ids(args.manifest, args.listening_cases)
    plan = {
        "format": FORMAT_VERSION,
        "purpose": "engineering_micro_overfit_not_heldout_evidence",
        "commands": commands,
        "train_dir": str(args.train_dir.resolve()),
        "eval_dir": str(args.eval_dir.resolve()),
        "checkpoint_each_epoch": True,
        "semantic_mixing_candidate": "question_residual",
        "health_profile": "micro_overfit",
        "listening_packet": {
            "selection_before_model_results": True,
            "selection_axes": "after_answerable,before_answerable,first_answerable,first_no_evidence_then_id_order",
            "requested_cases": args.listening_cases,
            "item_ids": listening_item_ids,
            "rendered_in_same_numeric_evaluation": bool(listening_item_ids),
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    state = gpu_state()
    require_idle_gpu(
        state,
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    print(json.dumps({"gpu_preflight": state}, sort_keys=True), flush=True)
    if args.evaluate_existing:
        if not (args.train_dir / "checkpoint.pt").is_file():
            raise SystemExit("--evaluate-existing requires train-dir/checkpoint.pt")
    else:
        if args.train_dir.exists() and any(args.train_dir.iterdir()):
            raise SystemExit(
                "train-dir is non-empty; refuse to mix or overwrite experiment outputs"
            )
        _run(commands["train"])

    if args.eval_dir.exists() and any(args.eval_dir.iterdir()):
        raise SystemExit(
            "eval-dir is non-empty; refuse to mix or overwrite evaluation outputs"
        )
    _run(commands["evaluate"])

    checkpoint = args.train_dir.resolve() / "checkpoint.pt"
    report = args.eval_dir.resolve() / "evaluation_report.json"
    try:
        health = build_health_receipt(checkpoint, report, "micro_overfit")
    except DemoContractError as error:
        raise SystemExit(str(error)) from error
    health_artifacts = write_health_outcome(args.train_dir, args.eval_dir, health)
    training_summary_path = args.train_dir.resolve() / "summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(report.read_text(encoding="utf-8"))
    analysis = analyze_microfit_result(training_summary, evaluation_payload, health)
    summary = {
        **plan,
        "gpu_preflight": state,
        "checkpoint": str(checkpoint),
        "evaluation_report": str(report),
        "training_summary": str(training_summary_path),
        **health_artifacts,
        "gates": health["gates"],
        "all_gates_passed": health["all_passed"],
        "permission": health["permission"],
        "failure_localization": analysis,
    }
    _atomic_json(args.eval_dir.resolve() / "microfit_gate_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not health["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
