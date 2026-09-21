"""Torch-free checkpoint gate and helpers for the interactive QCES demo."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


HEALTH_FORMAT = "qces_demo_checkpoint_health_v1"
GATE_PROFILES: Mapping[str, tuple[tuple[str, str, float], ...]] = {
    # This profile is only a memorization/sanity gate.  It must never be
    # presented as evidence of held-out generalization.
    "micro_overfit": (
        ("answerable_temporal_iou", ">=", 0.90),
        ("evidence_sd_sdri_answerable", ">", 0.0),
        ("no_evidence_balanced_accuracy", ">=", 0.90),
        ("no_evidence_auroc", ">=", 0.95),
        ("mean_no_evidence_retained_ratio", "<=", 0.10),
        ("maximum_mixture_consistency_l1", "<=", 1e-5),
    ),
    # The tIoU floor is deliberately above the frozen max-energy baseline
    # (0.251).  The waveform gate independently forbids a silent-mask win.
    "heldout_validation": (
        ("answerable_temporal_iou", ">=", 0.27),
        ("evidence_sd_sdri_answerable", ">", 0.0),
        ("no_evidence_balanced_accuracy", ">=", 0.60),
        ("no_evidence_auroc", ">=", 0.70),
        ("mean_no_evidence_retained_ratio", "<=", 0.10),
        ("maximum_mixture_consistency_l1", "<=", 1e-5),
    ),
}
METRIC_DIRECTIONS = {
    "answerable_temporal_iou": "↑",
    "evidence_sd_sdri_answerable": "↑",
    "no_evidence_balanced_accuracy": "↑",
    "no_evidence_auroc": "↑",
    "mean_no_evidence_retained_ratio": "↓",
    "maximum_mixture_consistency_l1": "↓",
}


class DemoContractError(ValueError):
    """Raised when demo inference would violate the checkpoint contract."""


def humanize_event_label(value: object) -> str:
    """Render one dataset/event label for a non-technical listening UI."""

    if not isinstance(value, str):
        return "unknown sound"
    cleaned = " ".join(value.replace("_", " ").strip().split())
    return cleaned or "unknown sound"


def relation_requirement(relation: object, question: object = "") -> str:
    """Explain the acoustic roles required without guessing an answer."""

    normalized = str(relation or "").strip().casefold()
    folded_question = str(question or "").casefold()
    if not normalized:
        if "before" in folded_question:
            normalized = "before"
        elif "after" in folded_question or "next" in folded_question:
            normalized = "after"
        elif "first" in folded_question or "sooner" in folded_question:
            normalized = "first"
    if normalized == "after":
        return (
            "Tìm âm mốc (anchor) được nhắc trong câu hỏi, rồi tìm sự kiện có "
            "onset ngay sau nó. Evidence phải giữ cả âm mốc và âm trả lời."
        )
    if normalized == "before":
        return (
            "Tìm âm mốc (anchor) được nhắc trong câu hỏi, rồi tìm sự kiện có "
            "onset ngay trước nó. Evidence phải giữ cả âm trả lời và âm mốc."
        )
    if normalized == "first":
        return (
            "Tìm hai âm ứng viên được nêu trong câu hỏi và so sánh onset. "
            "Evidence phải giữ cả hai để kiểm chứng âm nào bắt đầu trước."
        )
    return (
        "Tìm các đoạn âm thanh tối thiểu cần để kiểm chứng câu hỏi; không được "
        "suy ra evidence chỉ từ chữ trong câu hỏi."
    )


def compact_scene_events(events: object) -> list[dict[str, Any]]:
    """Return a chronological, display-safe event inventory from a manifest row."""

    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return []
    result: list[dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, Mapping):
            continue
        onset = raw.get("onset_seconds")
        offset = raw.get("offset_seconds")
        if (
            isinstance(onset, bool)
            or not isinstance(onset, (int, float))
            or isinstance(offset, bool)
            or not isinstance(offset, (int, float))
        ):
            continue
        occurrence = raw.get("occurrence_index", 1)
        result.append(
            {
                "event_id": str(raw.get("event_id", "")),
                "sound": humanize_event_label(raw.get("label")),
                "occurrence": (
                    int(occurrence)
                    if isinstance(occurrence, int) and not isinstance(occurrence, bool)
                    else 1
                ),
                "kind": str(raw.get("event_kind", "unknown")),
                "start_seconds": float(onset),
                "end_seconds": float(offset),
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["start_seconds"],
            item["end_seconds"],
            item["event_id"],
        ),
    )


def load_fingerprint_bound_examples(
    health: Mapping[str, Any], maximum_examples: int = 4
) -> list[dict[str, Any]]:
    """Load only pre-rendered examples from the health-bound evaluation manifest.

    Gold answers are returned for post-inference display, but callers must not put
    them in either model command.  Requiring the exact manifest hash and the
    report's pre-registered rendering IDs keeps this convenience path from
    silently turning into a hand-picked demo.
    """

    if maximum_examples <= 0:
        return []
    binding = health.get("evaluation_binding")
    report_identity = health.get("evaluation_report")
    if not isinstance(binding, Mapping) or not isinstance(report_identity, Mapping):
        raise DemoContractError("health receipt lacks evaluation bindings")
    manifest_text = binding.get("manifest")
    manifest_sha256 = binding.get("manifest_sha256")
    report_text = report_identity.get("path")
    if not isinstance(manifest_text, str) or not isinstance(manifest_sha256, str):
        raise DemoContractError("health receipt lacks a fingerprinted manifest")
    if not isinstance(report_text, str):
        raise DemoContractError("health receipt lacks its evaluation report path")
    manifest = Path(manifest_text).expanduser().resolve()
    if not manifest.is_file() or sha256_file(manifest) != manifest_sha256:
        raise DemoContractError("evaluation manifest changed after health evaluation")
    try:
        report = json.loads(Path(report_text).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoContractError(f"cannot read health-bound report: {error}") from error
    rendering = report.get("audio_rendering") if isinstance(report, dict) else None
    requested = (
        rendering.get("requested_item_ids") if isinstance(rendering, Mapping) else None
    )
    if not isinstance(requested, Sequence) or isinstance(requested, (str, bytes)):
        raise DemoContractError(
            "health-bound report lacks pre-registered listening IDs"
        )
    requested_ids = [item_id for item_id in requested if isinstance(item_id, str)]
    if not requested_ids:
        raise DemoContractError("health-bound report has no listening examples")

    rows: dict[str, dict[str, Any]] = {}
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise DemoContractError(
                        f"invalid example row at {manifest}:{line_number}"
                    )
                if row["id"] in rows:
                    raise DemoContractError(f"duplicate example ID: {row['id']}")
                rows[row["id"]] = row
    except (OSError, json.JSONDecodeError) as error:
        raise DemoContractError(f"cannot read evaluation manifest: {error}") from error

    examples: list[dict[str, Any]] = []
    for item_id in requested_ids[:maximum_examples]:
        row = rows.get(item_id)
        if row is None:
            raise DemoContractError(
                f"listening example is absent from manifest: {item_id}"
            )
        question = row.get("question")
        options = row.get("answer_options")
        answer = row.get("answer")
        answer_index = row.get("answer_option_index")
        mixture_text = row.get("mixture_path")
        if not isinstance(question, str) or not question.strip():
            raise DemoContractError(f"example has no question: {item_id}")
        if (
            not isinstance(options, list)
            or not 2 <= len(options) <= 5
            or not all(isinstance(option, str) and option.strip() for option in options)
        ):
            raise DemoContractError(f"example has invalid answer options: {item_id}")
        if (
            isinstance(answer_index, bool)
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < len(options)
            or not isinstance(answer, str)
            or options[answer_index] != answer
        ):
            raise DemoContractError(f"example has inconsistent gold answer: {item_id}")
        if not isinstance(mixture_text, str):
            raise DemoContractError(f"example has no mixture path: {item_id}")
        mixture = Path(mixture_text).expanduser()
        if not mixture.is_absolute():
            mixture = manifest.parent / mixture
        mixture = mixture.resolve()
        if not mixture.is_file():
            raise DemoContractError(f"example mixture does not exist: {mixture}")
        examples.append(
            {
                "id": item_id,
                "question": question.strip(),
                "answer_options": [option.strip() for option in options],
                "answer": answer,
                "answer_option_index": answer_index,
                "mixture_path": str(mixture),
                "relation": row.get("relation"),
                "no_evidence": bool(row.get("no_evidence")),
                "split": row.get("split"),
                # Scene annotations make the synthetic benchmark auditable in
                # the UI. Gold roles remain display-only and are never passed
                # to either inference command.
                "scene_events": compact_scene_events(row.get("events")),
                "anchor_event_ids": [
                    str(value)
                    for value in row.get("anchor_event_ids", [])
                    if isinstance(value, str)
                ],
                "answer_event_ids": [
                    str(value)
                    for value in row.get("answer_event_ids", [])
                    if isinstance(value, str)
                ],
                "anchor_intervals": row.get("anchor_intervals", []),
                "answer_intervals": row.get("answer_intervals", []),
            }
        )
    return examples


def gpu_resource_ready(
    state: Mapping[str, int] | None,
    minimum_free_mib: int,
    maximum_utilization_percent: int,
) -> bool:
    """Fail closed unless both free-memory and idle-utilization gates pass."""

    return bool(
        state is not None
        and state["free_memory_mib_↑"] >= minimum_free_mib
        and state["utilization_percent_↓"] <= maximum_utilization_percent
    )


def exclusive_lock_available(path: Path) -> bool:
    """Return whether a cross-process workflow lock can be acquired now.

    The probe releases the lock immediately.  Callers must acquire it again
    around the actual GPU operation because availability can change after UI
    rendering.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
    except OSError:
        return False
    return True


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DemoContractError(f"required file does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _finite_metric(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DemoContractError(f"evaluation summary lacks numeric metric: {key}")
    result = float(value)
    if not math.isfinite(result):
        raise DemoContractError(f"evaluation metric is non-finite: {key}")
    return result


def _passes(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    raise DemoContractError(f"unsupported gate operator: {operator}")


def build_health_receipt(
    checkpoint: Path, evaluation_report: Path, profile: str
) -> dict[str, Any]:
    if profile not in GATE_PROFILES:
        raise DemoContractError(f"unknown gate profile: {profile}")
    checkpoint_identity = file_identity(checkpoint)
    report_identity = file_identity(evaluation_report)
    try:
        report = json.loads(evaluation_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DemoContractError(f"cannot read evaluation report: {error}") from error
    if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
        raise DemoContractError("evaluation report is missing its summary object")
    declared_checkpoint = report.get("checkpoint")
    if not isinstance(declared_checkpoint, str):
        raise DemoContractError("evaluation report does not declare its checkpoint")
    if (
        Path(declared_checkpoint).expanduser().resolve()
        != checkpoint.expanduser().resolve()
    ):
        raise DemoContractError(
            "evaluation report was produced by a different checkpoint"
        )

    gates = []
    for key, operator, threshold in GATE_PROFILES[profile]:
        value = _finite_metric(report["summary"], key)
        gates.append(
            {
                "metric": key,
                "direction": METRIC_DIRECTIONS[key],
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": _passes(value, operator, threshold),
            }
        )
    return {
        "format": HEALTH_FORMAT,
        "profile": profile,
        "checkpoint": checkpoint_identity,
        "evaluation_report": report_identity,
        "evaluation_binding": {
            "declared_checkpoint": str(checkpoint.expanduser().resolve()),
            "manifest": report.get("manifest"),
            "manifest_sha256": report.get("manifest_sha256"),
            "report_format": report.get("format"),
            "schema_version": report.get("schema_version"),
        },
        "gates": gates,
        "all_passed": all(gate["passed"] for gate in gates),
        "permission": (
            "demo_inference_authorized"
            if all(gate["passed"] for gate in gates)
            else "demo_inference_blocked"
        ),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "contains_test_metric": False,
    }


def load_and_validate_health_receipt(
    checkpoint: Path, receipt_path: Path
) -> dict[str, Any]:
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DemoContractError(
            f"health receipt does not exist: {receipt_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise DemoContractError(f"cannot read health receipt: {error}") from error
    if not isinstance(receipt, dict) or receipt.get("format") != HEALTH_FORMAT:
        raise DemoContractError("unsupported checkpoint health receipt")
    actual_checkpoint = file_identity(checkpoint)
    if receipt.get("checkpoint") != actual_checkpoint:
        raise DemoContractError("checkpoint identity changed after health evaluation")
    report_entry = receipt.get("evaluation_report")
    if not isinstance(report_entry, dict) or not isinstance(
        report_entry.get("path"), str
    ):
        raise DemoContractError("health receipt lacks evaluation report identity")
    if file_identity(Path(report_entry["path"])) != report_entry:
        raise DemoContractError(
            "evaluation report changed after health receipt creation"
        )
    profile = receipt.get("profile")
    rebuilt = build_health_receipt(checkpoint, Path(report_entry["path"]), str(profile))
    comparable_keys = (
        "profile",
        "checkpoint",
        "evaluation_report",
        "evaluation_binding",
        "gates",
        "all_passed",
        "permission",
    )
    if any(receipt.get(key) != rebuilt.get(key) for key in comparable_keys):
        raise DemoContractError("health receipt contents do not reproduce")
    if not rebuilt["all_passed"]:
        raise DemoContractError("checkpoint failed at least one anti-collapse gate")
    return rebuilt
