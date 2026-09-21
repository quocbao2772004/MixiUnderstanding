#!/usr/bin/env python3
"""Run the fail-closed Phi-4 Multimodal one-record CUDA forward gate.

This gate establishes feasibility and deterministic exact-option scoring on the
15 GB T4 before a multi-record oracle calibration is allowed.  It is not an
accuracy result: one frozen validation record is scored under mixture, oracle
evidence, oracle residual, and question-only conditions, then mixture is scored
again in a fresh process to audit repeatability.
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
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

FORMAT = "qces_phi4mm_cuda_forward_gate_v1"
EXPECTED_AUDIOQA_FORMAT = "qces_audioqa_audit_v7"
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
REVISION = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"
PRIMARY_CONDITIONS = (
    "mixture",
    "oracle_evidence",
    "oracle_residual",
    "question_only",
)
SNAPSHOT_SHA256 = {
    "config.json": "49e1c05f93d43d7f17715b779a2576235b019f587285d7d914e5b05156253f62",
    "modeling_phi4mm.py": "e2b44eb7a66d6cc54524cee1ff9ba92d0658d435ea8900329ea0dbdb85c6439d",
    "processing_phi4mm.py": "84914d3e12256b4e2186e040c9830c11408468b6774f42afe85e6f8de2626d50",
    "speech_conformer_encoder.py": "3742827e945732cc5deea4a95e14004da037044431a94e3f3fac26239e614e3a",
    "tokenizer.json": "4c1b9f641d4f8b7247b8d5007dd3b6a9f6a87cb5123134fe0d326f14d10c0585",
    "model.safetensors.index.json": "b67dbc7062e1ccf472faba4222d631dc42929c827fbdaed1ec8e34fe0601819a",
    "model-00001-of-00003.safetensors": "c46bb03332d82f6a3eaf85bd20af388dd4d4d68b198c2203c965c7381a466094",
    "model-00002-of-00003.safetensors": "b3e812c0c8acef4e7f5e34d6c9f77a7640ee4a2b93ea351921365ac62f19918d",
    "model-00003-of-00003.safetensors": "7be96b7339303752634b202d3f377bcf312a03046586eca6cea23347ace1e65a",
    "speech-lora/adapter_config.json": "ed252a6ae210888ee69f5720bd7e8d8261f0abfda90b18ea6452316c71336df8",
    "speech-lora/adapter_model.safetensors": "1c2237461a4d1f9292cd128147bd3f0f70326a48d5d79c8e0f7583b26c095b30",
}
EXPECTED_PACKAGES = {
    "transformers": "4.48.2",
    "accelerate": "1.3.0",
    "peft": "0.13.2",
    "bitsandbytes": "0.45.2",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/qces-phi4mm/bin/python"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/qces_v5_paper/qces_val.jsonl",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "/home/cuongpv/.cache/huggingface/hub/"
            "models--microsoft--Phi-4-multimodal-instruct/snapshots/" + REVISION
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_phi4mm_forward_gate_seed2026",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "eager"),
        default="sdpa",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--minimum-free-gpu-mib", type=int, default=9_000)
    parser.add_argument("--maximum-gpu-utilization-percent", type=int, default=10)
    parser.add_argument("--repeatability-atol", type=float, default=1e-5)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0 or args.timeout_seconds <= 0:
        parser.error("bootstrap samples and timeout must be positive")
    if args.minimum_free_gpu_mib <= 0:
        parser.error("minimum free GPU memory must be positive")
    if not 0 <= args.maximum_gpu_utilization_percent <= 100:
        parser.error("maximum GPU utilization must lie in [0, 100]")
    if not math.isfinite(args.repeatability_atol) or args.repeatability_atol < 0:
        parser.error("repeatability tolerance must be finite and non-negative")
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def command_environment() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(CODE_ROOT) if not existing else str(CODE_ROOT) + os.pathsep + existing
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return environment


def runtime_packages(python: Path) -> dict[str, str]:
    code = (
        "import json; from importlib.metadata import version; "
        "names=('torch','transformers','torchvision','accelerate','peft',"
        "'bitsandbytes','numpy','scipy','soundfile'); "
        "print(json.dumps({name:version(name) for name in names},sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python.resolve()), "-c", code],
        cwd=PROJECT_ROOT,
        env=command_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot inspect Phi runtime: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("Phi runtime package inventory is invalid")
    return {str(name): str(version) for name, version in payload.items()}


def validate_assets(args: argparse.Namespace) -> dict[str, Any]:
    if not args.python.is_file():
        raise RuntimeError(f"Phi Python does not exist: {args.python.resolve()}")
    if not args.manifest.is_file():
        raise RuntimeError(
            f"validation manifest does not exist: {args.manifest.resolve()}"
        )
    snapshot = args.snapshot.resolve()
    if not snapshot.is_dir() or snapshot.name != REVISION:
        raise RuntimeError(f"wrong or missing Phi snapshot: {snapshot}")
    snapshot_files: dict[str, Any] = {}
    for relative, expected in SNAPSHOT_SHA256.items():
        path = snapshot / relative
        if not path.is_file():
            raise RuntimeError(f"Phi snapshot file is missing: {path}")
        identity = _identity(path)
        if identity["sha256"] != expected:
            raise RuntimeError(
                f"Phi snapshot hash mismatch for {relative}: "
                f"{identity['sha256']} != {expected}"
            )
        snapshot_files[relative] = identity
    packages = runtime_packages(args.python)
    for name, expected in EXPECTED_PACKAGES.items():
        if packages.get(name) != expected:
            raise RuntimeError(
                f"Phi runtime version mismatch for {name}: "
                f"{packages.get(name)} != {expected}"
            )
    if not packages.get("torch", "").startswith("2.6.0"):
        raise RuntimeError(
            f"Phi runtime has an unexpected torch: {packages.get('torch')}"
        )
    return {
        "python": _identity(args.python),
        "manifest": _identity(args.manifest),
        "snapshot": str(snapshot),
        "snapshot_files": snapshot_files,
        "runtime_packages": packages,
        "revision": REVISION,
        "model_id": MODEL_ID,
    }


def gpu_state() -> dict[str, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot query CUDA device: {completed.stderr.strip()}")
    try:
        free, utilization, temperature = [
            int(piece.strip())
            for piece in completed.stdout.strip().splitlines()[0].split(",")
        ]
    except (ValueError, IndexError) as error:
        raise RuntimeError("invalid nvidia-smi GPU state") from error
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


def wait_for_idle_gpu(
    minimum_free_mib: int,
    maximum_utilization: int,
    *,
    consecutive_samples: int = 3,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Wait for CUDA allocations/utilization to settle between fresh processes."""

    started = time.monotonic()
    streak = 0
    samples: list[dict[str, int]] = []
    while time.monotonic() - started <= timeout_seconds:
        state = gpu_state()
        samples.append(state)
        try:
            require_idle_gpu(state, minimum_free_mib, maximum_utilization)
        except RuntimeError:
            streak = 0
        else:
            streak += 1
            if streak >= consecutive_samples:
                return {
                    "ready": True,
                    "consecutive_idle_samples_↑": streak,
                    "elapsed_seconds_↓": time.monotonic() - started,
                    "final_state": state,
                    "sample_count_↑": len(samples),
                }
        time.sleep(1.0)
    raise RuntimeError(
        "GPU did not settle between Phi stages; final samples: " + str(samples[-3:])
    )


def build_command(
    args: argparse.Namespace,
    output_dir: Path,
    conditions: Sequence[str],
    *,
    validate_only: bool = False,
) -> list[str]:
    command = [
        str(args.python.resolve()),
        str(CODE_ROOT / "mixi_understanding/scripts/evaluate_qces_audioqa.py"),
        "--auditor",
        "phi4mm",
        "--manifest",
        str(args.manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--model",
        MODEL_ID,
        "--revision",
        REVISION,
        "--conditions",
        *conditions,
        "--split",
        "val",
        "--max-records",
        "1",
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


def _pid_gpu_memory_mib(pid: int) -> int | None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 2:
            continue
        try:
            if int(pieces[0]) == pid:
                return int(pieces[1])
        except ValueError:
            continue
    return 0


def run_monitored(
    command: Sequence[str], log_prefix: Path, timeout_seconds: int
) -> dict[str, Any]:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_prefix.with_suffix(".stdout.txt")
    stderr_path = log_prefix.with_suffix(".stderr.txt")
    started = time.monotonic()
    peak_mib = 0
    samples = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=command_environment(),
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise RuntimeError(
                    f"Phi stage timed out after {timeout_seconds}s; see {stderr_path}"
                )
            used = _pid_gpu_memory_mib(process.pid)
            if used is not None:
                peak_mib = max(peak_mib, used)
                samples += 1
            time.sleep(0.25)
        returncode = int(process.returncode)
    elapsed = time.monotonic() - started
    result = {
        "command": list(command),
        "returncode": returncode,
        "elapsed_seconds_↓": elapsed,
        "peak_process_gpu_memory_mib_↓": peak_mib,
        "gpu_memory_samples_↑": samples,
        "stdout": _identity(stdout_path),
        "stderr": _identity(stderr_path),
    }
    monitor_path = log_prefix.with_suffix(".monitor.json")
    _atomic_json(monitor_path, result)
    if returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-4_000:]
        raise RuntimeError(
            f"Phi evaluator exited {returncode}; see {stderr_path}\n{tail}"
        )
    if peak_mib <= 0 or samples <= 0:
        raise RuntimeError("Phi forward finished without observable process GPU memory")
    return result


def load_monitor(log_prefix: Path, expected_command: Sequence[str]) -> dict[str, Any]:
    path = log_prefix.with_suffix(".monitor.json")
    if not path.is_file():
        raise RuntimeError(
            "completed Phi output lacks runner-owned GPU monitoring: " + str(path)
        )
    payload = _read_json(path)
    if payload.get("command") != list(expected_command):
        raise RuntimeError(
            "Phi monitor command does not match the frozen stage command"
        )
    if payload.get("returncode") != 0:
        raise RuntimeError("Phi monitor records a failed evaluator process")
    peak = payload.get("peak_process_gpu_memory_mib_↓")
    samples = payload.get("gpu_memory_samples_↑")
    if (
        isinstance(peak, bool)
        or not isinstance(peak, (int, float))
        or peak <= 0
        or isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples <= 0
    ):
        raise RuntimeError("Phi monitor lacks positive process-GPU observations")
    return payload


def _load_items(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError(f"invalid Phi item at {path}:{line_number}")
            items.append(payload)
    return items


def validate_output(
    output_dir: Path, expected_conditions: Sequence[str]
) -> dict[str, Any]:
    metadata_path = output_dir / "run_metadata.json"
    report_path = output_dir / "evaluation_report.json"
    items_path = output_dir / "items.jsonl"
    for path in (metadata_path, report_path, items_path):
        if not path.is_file():
            raise RuntimeError(f"Phi output is incomplete: {path}")
    metadata = _read_json(metadata_path)
    report = _read_json(report_path)
    run_config = metadata.get("run_config")
    runtime = metadata.get("runtime_model")
    if not isinstance(run_config, Mapping) or not isinstance(runtime, Mapping):
        raise RuntimeError("Phi metadata lacks run/runtime configuration")
    expected = (
        (run_config.get("auditor") == "phi4mm", "wrong auditor"),
        (run_config.get("model") == MODEL_ID, "wrong model"),
        (run_config.get("revision") == REVISION, "wrong requested revision"),
        (run_config.get("quantization") == "4bit", "wrong quantization"),
        (run_config.get("dtype") == "float16", "wrong compute dtype"),
        (run_config.get("device") == "cuda", "wrong device"),
        (run_config.get("max_records") == 1, "wrong record count"),
        (
            list(run_config.get("conditions", [])) == list(expected_conditions),
            "wrong conditions",
        ),
        (runtime.get("auditor_family") == "phi4_multimodal", "wrong architecture"),
        (runtime.get("requested_revision") == REVISION, "runtime revision mismatch"),
        (runtime.get("resolved_commit_hash") == REVISION, "resolved commit mismatch"),
        (runtime.get("trusted_custom_code") is True, "custom code was not declared"),
        (
            runtime.get("option_continuation_token_ids")
            == [[32], [33], [34], [35], [36]],
            "option letters are not the audited one-token IDs",
        ),
    )
    errors = [message for passed, message in expected if not passed]
    if report.get("format") != EXPECTED_AUDIOQA_FORMAT:
        errors.append("wrong AudioQA report format")
    items = _load_items(items_path)
    if len(items) != len(expected_conditions):
        errors.append("wrong completed condition count")
    if {item.get("condition") for item in items} != set(expected_conditions):
        errors.append("item condition coverage mismatch")
    record_ids = {item.get("id") for item in items}
    if len(record_ids) != 1:
        errors.append("forward gate did not use exactly one record")
    for item in items:
        scores = item.get("option_log_scores")
        probabilities = item.get("option_probabilities")
        if (
            not isinstance(scores, list)
            or len(scores) != 5
            or not all(
                isinstance(value, (int, float)) and math.isfinite(float(value))
                for value in scores
            )
        ):
            errors.append(f"non-finite/wrong option scores: {item.get('condition')}")
        if (
            not isinstance(probabilities, list)
            or len(probabilities) != 5
            or not all(
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) >= 0
                for value in probabilities
            )
            or not math.isclose(sum(probabilities or []), 1.0, abs_tol=1e-5)
        ):
            errors.append(f"invalid probabilities: {item.get('condition')}")
        if item.get("candidate_token_lengths") != [1, 1, 1, 1, 1]:
            errors.append(f"multi-token option scoring: {item.get('condition')}")
        if item.get("scoring_method") != "single_token_next_log_probability":
            errors.append(f"wrong scoring method: {item.get('condition')}")
    if report.get("run_fingerprint") != metadata.get("run_fingerprint"):
        errors.append("report/metadata run fingerprint mismatch")
    if errors:
        raise RuntimeError("invalid Phi forward output: " + "; ".join(errors))
    return {
        "output_dir": str(output_dir.resolve()),
        "metadata": _identity(metadata_path),
        "report": _identity(report_path),
        "items": _identity(items_path),
        "run_fingerprint": metadata["run_fingerprint"],
        "record_id": next(iter(record_ids)),
        "runtime_model": dict(runtime),
        "condition_items": {item["condition"]: item for item in items},
    }


def compare_repeatability(
    primary: Mapping[str, Any], repeat: Mapping[str, Any], atol: float
) -> dict[str, Any]:
    first = primary["condition_items"]["mixture"]
    second = repeat["condition_items"]["mixture"]
    scores_a = [float(value) for value in first["option_log_scores"]]
    scores_b = [float(value) for value in second["option_log_scores"]]
    maximum_difference = max(
        abs(left - right) for left, right in zip(scores_a, scores_b)
    )
    checks = {
        "same_record_id": primary["record_id"] == repeat["record_id"],
        "same_source_audio_sha256": first.get("source_audio_sha256")
        == second.get("source_audio_sha256"),
        "same_question": first.get("question") == second.get("question"),
        "same_presented_options": first.get("answer_options")
        == second.get("answer_options"),
        "same_prediction": first.get("predicted_answer")
        == second.get("predicted_answer"),
        "maximum_log_score_absolute_difference_within_tolerance": maximum_difference
        <= atol,
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "maximum_log_score_absolute_difference_↓": maximum_difference,
        "absolute_tolerance_↓": atol,
    }


def _run_validate_only(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=command_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Phi evaluator validate-only failed: "
            + (completed.stderr or completed.stdout)[-4_000:]
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict) or payload.get("records") != 1:
        raise RuntimeError("Phi validate-only returned an invalid record count")
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    assets = validate_assets(args)
    root = args.results_root.resolve()
    primary_dir = root / "primary"
    repeat_dir = root / "repeat_mixture"
    primary_command = build_command(args, primary_dir, PRIMARY_CONDITIONS)
    repeat_command = build_command(args, repeat_dir, ("mixture",))
    validation = _run_validate_only(
        build_command(args, primary_dir, PRIMARY_CONDITIONS, validate_only=True)
    )
    dry_run = {
        "format": FORMAT,
        "mode": "execute" if args.execute else "dry_run",
        "assets": assets,
        "validate_only": validation,
        "primary_command": primary_command,
        "repeat_command": repeat_command,
        "predeclared_gates": {
            "finite_five_option_logits_↑": True,
            "one_token_option_ids_↑": [[32], [33], [34], [35], [36]],
            "resolved_commit_match_↑": REVISION,
            "observable_process_gpu_memory_↑": True,
            "repeat_prediction_match_↑": True,
            "maximum_log_score_absolute_difference_↓": args.repeatability_atol,
            "accuracy_gate_predeclared": False,
        },
    }
    if not args.execute:
        print(json.dumps(dry_run, indent=2, sort_keys=True))
        return

    state = gpu_state()
    require_idle_gpu(
        state, args.minimum_free_gpu_mib, args.maximum_gpu_utilization_percent
    )
    root.mkdir(parents=True, exist_ok=True)
    primary_log = root / "logs/primary"
    repeat_log = root / "logs/repeat_mixture"
    if (primary_dir / "evaluation_report.json").is_file():
        primary = validate_output(primary_dir, PRIMARY_CONDITIONS)
        primary_monitor = load_monitor(primary_log, primary_command)
    else:
        primary_monitor = run_monitored(
            primary_command, primary_log, args.timeout_seconds
        )
        primary = validate_output(primary_dir, PRIMARY_CONDITIONS)
    between_stage_idle = wait_for_idle_gpu(
        args.minimum_free_gpu_mib,
        args.maximum_gpu_utilization_percent,
    )
    if (repeat_dir / "evaluation_report.json").is_file():
        repeat = validate_output(repeat_dir, ("mixture",))
        repeat_monitor = load_monitor(repeat_log, repeat_command)
    else:
        repeat_monitor = run_monitored(repeat_command, repeat_log, args.timeout_seconds)
        repeat = validate_output(repeat_dir, ("mixture",))
    repeatability = compare_repeatability(primary, repeat, args.repeatability_atol)
    if not repeatability["all_checks_passed"]:
        raise RuntimeError(f"Phi repeatability gate failed: {repeatability}")
    receipt = {
        **dry_run,
        "format": FORMAT,
        "mode": "completed",
        "gpu_preflight": state,
        "primary": {
            key: value for key, value in primary.items() if key != "condition_items"
        },
        "repeat": {
            key: value for key, value in repeat.items() if key != "condition_items"
        },
        "monitoring": {"primary": primary_monitor, "repeat": repeat_monitor},
        "between_stage_idle_gate": between_stage_idle,
        "repeatability": repeatability,
        "all_gates_passed": True,
        "permission": "phi4mm_oracle_calibration_authorized",
        "paper_accuracy_result": False,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    receipt_path = root / "phi4mm_forward_gate_receipt.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"wrote {receipt_path}")


if __name__ == "__main__":
    main()
