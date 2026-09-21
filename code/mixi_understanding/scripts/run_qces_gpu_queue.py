#!/usr/bin/env python3
"""Durable, receipt-driven GPU queue for the QCES development gates.

The queue waits for stable idle GPU samples and resumes each registered runner
from its authoritative receipt.  It deliberately stops after the one-seed
ablation screen: the 21-run full matrix requires analysis of that screen before
its frozen registry may be executed.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
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
from mixi_understanding.scripts.run_qces_ablation_dev_screen import (  # noqa: E402
    FORMAT as ABLATION_FORMAT,
)
from mixi_understanding.scripts.run_qces_cee_dev_pilot import (  # noqa: E402
    FORMAT as CEE_FORMAT,
)
from mixi_understanding.scripts.run_qces_cee_memory_smoke import (  # noqa: E402
    FORMAT as CEE_MEMORY_FORMAT,
)
from mixi_understanding.scripts.run_qces_caption_planner_devpilot import (  # noqa: E402
    FORMAT as CAPTION_BASELINE_FORMAT,
)
from mixi_understanding.scripts.run_qces_phi4mm_forward_gate import (  # noqa: E402
    FORMAT as PHI_FORWARD_FORMAT,
)
from mixi_understanding.scripts.run_qces_phi4mm_oracle_calibration import (  # noqa: E402
    FORMAT as PHI_ORACLE_FORMAT,
)


FORMAT = "qces_durable_gpu_queue_v1"
STAGES = (
    "microfit",
    "cee_memory",
    "cee_pilot",
    "phi_forward",
    "phi_oracle",
    "caption_baseline",
    "ablation_screen",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/comfyui/bin/python3.10"),
    )
    parser.add_argument(
        "--phi-python",
        type=Path,
        default=Path("/home/cuongpv/anaconda3/envs/qces-phi4mm/bin/python"),
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_gpu_queue_status.json",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_gpu_queue.log",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=PROJECT_ROOT / "outputs/.qces_gpu_queue.lock",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--idle-streak", type=int, default=3)
    parser.add_argument("--stop-after", choices=STAGES, default="ablation_screen")
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or args.poll_seconds > 60:
        parser.error("poll seconds must lie in [1, 60]")
    if args.idle_streak <= 0:
        parser.error("idle streak must be positive")
    return args


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.resolve().read_text("utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Queue:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.completed: list[str] = []
        self.skipped: dict[str, str] = {}
        self.current_stage = "startup"
        self.last_gpu: dict[str, int] | None = None
        self.started_at = _utc_now()

    def emit(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line, flush=True)
        self.args.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.args.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def status(self, state: str, *, error: str | None = None) -> None:
        _atomic_json(
            self.args.status_path,
            {
                "format": FORMAT,
                "state": state,
                "started_at_utc": self.started_at,
                "updated_at_utc": _utc_now(),
                "current_stage": self.current_stage,
                "completed_stages": list(self.completed),
                "skipped_stages": dict(self.skipped),
                "last_gpu_sample": self.last_gpu,
                "error": error,
                "stop_after": self.args.stop_after,
                "metric_direction_legend": {
                    "↑": "higher is better",
                    "↓": "lower is better",
                },
            },
        )

    def gpu_state(self) -> dict[str, int]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        free, utilization, temperature = [
            int(item.strip())
            for item in completed.stdout.strip().splitlines()[0].split(",")
        ]
        self.last_gpu = {
            "free_memory_mib_↑": free,
            "utilization_percent_↓": utilization,
            "temperature_celsius_↓": temperature,
        }
        return self.last_gpu

    def wait_idle(self, *, minimum_free: int, maximum_utilization: int) -> None:
        streak = 0
        while streak < self.args.idle_streak:
            try:
                state = self.gpu_state()
                ready = (
                    state["free_memory_mib_↑"] >= minimum_free
                    and state["utilization_percent_↓"] <= maximum_utilization
                )
                streak = streak + 1 if ready else 0
                self.emit(
                    f"{self.current_stage}: free={state['free_memory_mib_↑']} MiB ↑ "
                    f"util={state['utilization_percent_↓']}% ↓ "
                    f"temp={state['temperature_celsius_↓']}C ↓ "
                    f"idle_streak={streak}/{self.args.idle_streak}"
                )
                self.status("waiting_for_idle_gpu")
            except (
                OSError,
                ValueError,
                IndexError,
                subprocess.SubprocessError,
            ) as error:
                streak = 0
                self.emit(f"{self.current_stage}: GPU query unavailable: {error}")
                self.status("waiting_for_gpu_device", error=str(error))
            if streak < self.args.idle_streak:
                time.sleep(self.args.poll_seconds)

    def run_command(self, label: str, command: Sequence[str]) -> None:
        self.emit(f"launching {label}: {json.dumps(list(command))}")
        self.status("running")
        environment = os.environ.copy()
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        environment["PYTHONPATH"] = str(CODE_ROOT) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
        with self.args.log_path.open("a", encoding="utf-8") as handle:
            subprocess.run(
                list(command),
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        self.emit(f"completed {label}")

    def finish_stage(self, stage: str) -> bool:
        self.completed.append(stage)
        self.status("stage_completed")
        return stage == self.args.stop_after


def _valid_json_format(path: Path, expected: str) -> bool:
    try:
        return _read_json(path).get("format") == expected
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False


def microfit_state() -> tuple[str, str | None]:
    checkpoint = PROJECT_ROOT / (
        "outputs/qces_v5_microfit_seed2028/union_train_balanced_v3/checkpoint.pt"
    )
    health = PROJECT_ROOT / (
        "outputs/qces_v5_microfit_seed2028/union_train_balanced_v3/"
        "demo_health_receipt.json"
    )
    failed = PROJECT_ROOT / (
        "outputs/qces_v5_microfit_seed2028/union_eval_balanced_v3/"
        "failed_health_audit.json"
    )
    if health.is_file():
        try:
            receipt = load_and_validate_health_receipt(checkpoint, health)
            if receipt.get("permission") == "demo_inference_authorized":
                return "passed", None
            return "failed", "microfit receipt lacks demo permission"
        except (DemoContractError, OSError) as error:
            return "failed", str(error)
    if failed.is_file():
        return "failed", str(failed)
    return "pending", None


def cee_promotion(path: Path) -> tuple[bool, str]:
    try:
        payload = _read_json(path)
        passed = bool(
            payload.get("format") == CEE_FORMAT
            and payload.get("integrity", {}).get("all_passed") is True
            and payload.get("decision", {}).get("promote_cee_to_full_seeded_run")
            is True
        )
        return passed, "promoted" if passed else "CEE pilot did not promote"
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        return False, str(error)


def run_queue(args: argparse.Namespace) -> None:
    queue = Queue(args)
    queue.status("starting")
    scripts = CODE_ROOT / "mixi_understanding/scripts"

    queue.current_stage = "microfit"
    state, reason = microfit_state()
    if state == "pending":
        queue.wait_idle(minimum_free=7_000, maximum_utilization=10)
        try:
            queue.run_command(
                "microfit anti-collapse gate",
                [str(args.python), str(scripts / "run_qces_microfit_gate.py")],
            )
        except subprocess.CalledProcessError:
            state, reason = microfit_state()
            if state == "pending":
                raise
        state, reason = microfit_state()
    if state == "passed":
        if queue.finish_stage("microfit"):
            return
    else:
        queue.skipped["microfit"] = reason or "microfit failed"
        queue.skipped["cee_memory"] = "microfit prerequisite failed"
        queue.skipped["cee_pilot"] = "microfit prerequisite failed"
        queue.skipped["ablation_screen"] = "CEE pilot unavailable"
        queue.emit(f"microfit failed; preserving audit and skipping CEE: {reason}")

    if state == "passed":
        queue.current_stage = "cee_memory"
        memory_receipt = PROJECT_ROOT / (
            "outputs/qces_v5_cee_memory_smoke_seed10/amp_batch3/"
            "cee_memory_smoke_receipt.json"
        )
        if not _valid_json_format(memory_receipt, CEE_MEMORY_FORMAT):
            queue.wait_idle(minimum_free=9_000, maximum_utilization=10)
            queue.run_command(
                "CEE AMP batch-3 memory smoke",
                [str(args.python), str(scripts / "run_qces_cee_memory_smoke.py")],
            )
        receipt = _read_json(memory_receipt)
        if receipt.get("gates", {}).get("all_gates_passed") is not True:
            raise RuntimeError("CEE memory receipt did not pass all gates")
        if queue.finish_stage("cee_memory"):
            return

        queue.current_stage = "cee_pilot"
        comparison = (
            PROJECT_ROOT / "outputs/qces_v5_cee_devpilot_seed2026/comparison.json"
        )
        if not _valid_json_format(comparison, CEE_FORMAT):
            queue.wait_idle(minimum_free=9_000, maximum_utilization=10)
            queue.run_command(
                "CEE schedule-matched held-out pilot",
                [
                    str(args.python),
                    str(scripts / "run_qces_cee_dev_pilot.py"),
                    "--execute",
                ],
            )
        if queue.finish_stage("cee_pilot"):
            return

    queue.current_stage = "phi_forward"
    forward_receipt = PROJECT_ROOT / (
        "outputs/qces_v5_phi4mm_forward_gate_seed2026/"
        "phi4mm_forward_gate_receipt.json"
    )
    if not _valid_json_format(forward_receipt, PHI_FORWARD_FORMAT):
        queue.wait_idle(minimum_free=9_000, maximum_utilization=10)
        queue.run_command(
            "Phi-4MM forward and repeatability gate",
            [
                str(args.phi_python),
                str(scripts / "run_qces_phi4mm_forward_gate.py"),
                "--execute",
            ],
        )
    forward = _read_json(forward_receipt)
    if forward.get("all_gates_passed") is not True:
        raise RuntimeError("Phi forward receipt did not pass all gates")
    if queue.finish_stage("phi_forward"):
        return

    queue.current_stage = "phi_oracle"
    oracle_receipt = PROJECT_ROOT / (
        "outputs/qces_v5_phi4mm_oracle_calibration_seed2026/"
        "phi4mm_oracle_calibration_receipt.json"
    )
    if not _valid_json_format(oracle_receipt, PHI_ORACLE_FORMAT):
        queue.wait_idle(minimum_free=9_000, maximum_utilization=10)
        queue.run_command(
            "Phi-4MM cross-family oracle calibration",
            [
                str(args.phi_python),
                str(scripts / "run_qces_phi4mm_oracle_calibration.py"),
                "--execute",
            ],
        )
    if queue.finish_stage("phi_oracle"):
        return

    queue.current_stage = "caption_baseline"
    caption_receipt = PROJECT_ROOT / (
        "outputs/qces_v5_caption_planner_audiosep_devpilot_seed2026/"
        "devpilot_comparison_receipt.json"
    )
    if not _valid_json_format(caption_receipt, CAPTION_BASELINE_FORMAT):
        queue.wait_idle(minimum_free=9_000, maximum_utilization=10)
        try:
            queue.run_command(
                "AF3 caption/planner to frozen AudioSep development baseline",
                [
                    str(args.python),
                    str(scripts / "run_qces_caption_planner_devpilot.py"),
                    "--execute",
                ],
            )
        except subprocess.CalledProcessError as error:
            queue.skipped["caption_baseline"] = (
                f"runner exited with code {error.returncode}; inspect durable log"
            )
            queue.emit(
                "caption/planner baseline failed; preserving its resumable outputs "
                "and continuing to the independent ablation screen"
            )
    if _valid_json_format(caption_receipt, CAPTION_BASELINE_FORMAT):
        caption = _read_json(caption_receipt)
        if caption.get("integrity", {}).get("all_passed") is not True:
            queue.skipped["caption_baseline"] = "comparison receipt failed integrity"
        elif queue.finish_stage("caption_baseline"):
            return
    elif "caption_baseline" not in queue.skipped:
        queue.skipped["caption_baseline"] = "comparison receipt unavailable"

    queue.current_stage = "ablation_screen"
    comparison = PROJECT_ROOT / "outputs/qces_v5_cee_devpilot_seed2026/comparison.json"
    promoted, reason = cee_promotion(comparison)
    if not promoted:
        queue.skipped["ablation_screen"] = reason
        queue.emit(f"ablation screen skipped: {reason}")
        queue.status("completed_with_skips")
        return
    ablation = PROJECT_ROOT / (
        "outputs/qces_v5_ablation_dev_screen_seed2026/"
        "ablation_screen_comparison.json"
    )
    if not _valid_json_format(ablation, ABLATION_FORMAT):
        queue.wait_idle(minimum_free=12_000, maximum_utilization=10)
        queue.run_command(
            "one-seed contribution ablation screen",
            [
                str(args.python),
                str(scripts / "run_qces_ablation_dev_screen.py"),
                "--execute",
                "--minimum-free-gpu-mib",
                "12000",
            ],
        )
    queue.finish_stage("ablation_screen")
    queue.emit(
        "queue stopped after ablation screen; analyze results before freezing "
        "or executing the 21-run full matrix"
    )
    queue.status("completed_analysis_required")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with args.lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("another QCES GPU queue owns the lock") from error
        try:
            run_queue(args)
        except Exception as error:
            queue_status = {
                "format": FORMAT,
                "state": "failed",
                "updated_at_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
            _atomic_json(args.status_path, queue_status)
            with args.log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{_utc_now()} queue failed: {type(error).__name__}: {error}\n"
                )
            raise


if __name__ == "__main__":
    main()
