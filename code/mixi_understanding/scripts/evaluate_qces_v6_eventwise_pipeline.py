#!/usr/bin/env python3
"""Evaluate QCES-v6 with an event-wise evidence renderer.

This file is intentionally copied from ``evaluate_qces_v6_pipeline.py`` instead
of editing that script.  It keeps the original QCES-v6 receipt evaluator intact
and adds deployable event-wise rendering modes:

``eventwise__predicted_gate``
    Plan the same events as the current deployable system, but query AudioSep
    with one canonical prompt per planned event label and gate every prompt by
    that event's own span before summing the stems.

``eventwise__predicted_no_gate``
    Same event-wise prompts without temporal gating.  This diagnostic separates
    semantic leakage from temporal-gate leakage.

The goal is to reduce the leakage heard in joint prompts such as
``"the sounds of bark and meow"`` by rendering ``"bark"`` and ``"meow"``
separately, then composing the answer evidence.

The deployable mode, ``predicted``, reads the question surface string, the
answer options and the mixture waveform.  It never reads the annotated event
inventory, the gold answer, the gold evidence IDs or the target waveform.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.event_proposals import (
    EventProposal,
    ProposalHead,
    decode_proposals,
    energy_activity,
    features_from_cache,
    zero_feature_groups,
)
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.relational_planner import (
    EvidencePlan,
    plan_from_proposals,
    prompt_for_labels,
)
from mixi_understanding.qces.stem_features import FrameGrid
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    _v5_item_metrics,
    describe,
    encode_prompts,
    oracle_prompt,
    summarize_v5_items,
    write_item_artifacts,
)

FORMAT_VERSION = "qces_v6_eventwise_pipeline_evaluation_v1"

MODE_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    "mixture_passthrough": {
        "description": "Return the mixture unchanged; the zero-improvement reference.",
        "reads_annotation": "none",
    },
    "question_prompt__no_gate": {
        "description": "Frozen separator queried with the raw question string.",
        "reads_annotation": "none",
    },
    "predicted__no_gate": {
        "description": "Parsed question and predicted proposals choose the prompt; no temporal gate.",
        "reads_annotation": "none",
    },
    "predicted__predicted_gate": {
        "description": "Full QCES-v6 system: parsed question, predicted proposals, predicted spans.",
        "reads_annotation": "none",
    },
    "eventwise__predicted_gate": {
        "description": (
            "Event-wise renderer: parsed question and predicted proposals choose "
            "the evidence events; AudioSep is queried once per planned event "
            "label and each stem is gated by that event's predicted span before "
            "summing."
        ),
        "reads_annotation": "none",
    },
    "eventwise__predicted_no_gate": {
        "description": (
            "Event-wise semantic diagnostic: parsed question and predicted "
            "proposals choose events; AudioSep is queried per event label, but "
            "the per-label stems are summed without temporal gates."
        ),
        "reads_annotation": "none",
    },
    "predicted__best_effort": {
        "description": (
            "Deployable variant that never returns silence: when the planner "
            "abstains, the separator is still queried with the labels the "
            "question names, and the no-evidence decision is reported "
            "separately instead of being applied to the waveform."
        ),
        "reads_annotation": "none",
    },
    "energy_predicted__predicted_gate": {
        "description": "Same pipeline with the training-free energy proposal reader.",
        "reads_annotation": "none",
    },
    "oracle_inventory__planned_gate": {
        "description": "Parsed question planned over the annotated event inventory.",
        "reads_annotation": "event_inventory",
    },
    "oracle_text__no_gate": {
        "description": "Annotated evidence labels as the prompt, no temporal gate.",
        "reads_annotation": "evidence_labels",
    },
    "oracle_text__oracle_gate": {
        "description": "Annotated evidence labels and annotated evidence spans; the upper bound.",
        "reads_annotation": "evidence_labels_and_spans",
    },
}

DEFAULT_MODES = (
    "mixture_passthrough",
    "question_prompt__no_gate",
    "predicted__no_gate",
    "predicted__predicted_gate",
    "eventwise__predicted_gate",
    "eventwise__predicted_no_gate",
    "predicted__best_effort",
    "energy_predicted__predicted_gate",
    "oracle_inventory__planned_gate",
    "oracle_text__no_gate",
    "oracle_text__oracle_gate",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--stem-cache", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--relation-threshold",
        action="append",
        default=[],
        metavar="REL=VALUE",
        help=(
            "Override learned proposal decoding threshold for a relation, e.g. "
            "--relation-threshold after=0.12 --relation-threshold before=0.15. "
            "Only deployable learned-proposal modes use these values."
        ),
    )
    parser.add_argument("--energy-threshold", type=float, default=0.3)
    parser.add_argument("--gate-dilation-seconds", type=float, default=0.0)
    parser.add_argument(
        "--eventwise-padding-seconds",
        type=float,
        default=0.05,
        help="Symmetric padding around every event-wise span before gating.",
    )
    parser.add_argument(
        "--eventwise-fade-seconds",
        type=float,
        default=0.02,
        help="Linear fade length applied at event-wise gate boundaries.",
    )
    parser.add_argument(
        "--render-mode",
        action="append",
        default=[],
        help=(
            "Write predicted_evidence.wav / predicted_residual.wav for this mode "
            "in the scene/question layout the frozen-auditor script consumes."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.modes) - set(MODE_REGISTRY))
    if unknown:
        parser.error("unknown modes: " + ", ".join(unknown))
    return args


def parse_relation_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    allowed = {"after", "before", "first"}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --relation-threshold {value!r}; expected REL=VALUE")
        relation, raw = value.split("=", 1)
        relation = relation.strip()
        if relation not in allowed:
            raise SystemExit(
                f"invalid relation threshold {relation!r}; expected one of {sorted(allowed)}"
            )
        try:
            cut = float(raw)
        except ValueError as exc:
            raise SystemExit(f"invalid threshold value in {value!r}") from exc
        if not 0.0 <= cut <= 1.0:
            raise SystemExit(f"threshold out of range in {value!r}")
        thresholds[relation] = cut
    return thresholds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_taxonomy(path: Path) -> tuple[str, ...]:
    composition = json.loads(path.read_text(encoding="utf-8"))["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


@dataclass(frozen=True)
class Request:
    """One separator query plus the temporal gate applied to its output."""

    prompt: str
    spans: tuple[tuple[float, float], ...]
    gated: bool
    silent: bool


@dataclass(frozen=True)
class EventwiseRequest:
    """Several single-label separator queries composed into one evidence stem."""

    events: tuple[EventProposal, ...]
    gated: bool
    silent: bool

    @property
    def spans(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (item.onset_seconds, item.offset_seconds) for item in self.events
        )


def request_prompt_text(request: Request | EventwiseRequest) -> str:
    if isinstance(request, Request):
        return request.prompt
    parts = [
        f"{item.label}@{item.onset_seconds:.2f}-{item.offset_seconds:.2f}"
        for item in request.events
    ]
    return "eventwise: " + "; ".join(parts) if parts else ""


def request_prompts(
    request: Request | EventwiseRequest, describe_fn
) -> tuple[str, ...]:
    if isinstance(request, Request):
        return (request.prompt,) if request.prompt and not request.silent else ()
    if request.silent:
        return ()
    return tuple(
        dict.fromkeys(
            prompt_for_labels((item.label,), describe_fn) for item in request.events
        )
    )


def spans_by_label(
    events: Sequence[EventProposal],
) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in events:
        grouped[item.label].append((item.onset_seconds, item.offset_seconds))
    return dict(grouped)


def span_gate(
    spans: Sequence[tuple[float, float]],
    num_samples: int,
    sample_rate: int,
    device: torch.device,
    dilation_seconds: float,
) -> torch.Tensor:
    gate = torch.zeros(num_samples, device=device)
    dilation = int(round(dilation_seconds * sample_rate))
    for onset, offset in spans:
        start = max(0, min(num_samples, int(round(onset * sample_rate)) - dilation))
        stop = max(start, min(num_samples, int(round(offset * sample_rate)) + dilation))
        gate[start:stop] = 1.0
    return gate


def span_gate_with_fade(
    spans: Sequence[tuple[float, float]],
    num_samples: int,
    sample_rate: int,
    device: torch.device,
    padding_seconds: float,
    fade_seconds: float,
) -> torch.Tensor:
    """Union gate with optional padding and boundary fades.

    This is deliberately used only by the new event-wise modes, so the original
    QCES-v6 gate metric remains exactly reproducible.
    """

    gate = torch.zeros(num_samples, device=device)
    padding = max(0, int(round(padding_seconds * sample_rate)))
    fade = max(0, int(round(fade_seconds * sample_rate)))
    for onset, offset in spans:
        start = max(0, min(num_samples, int(round(onset * sample_rate)) - padding))
        stop = max(start, min(num_samples, int(round(offset * sample_rate)) + padding))
        if stop <= start:
            continue
        local = torch.ones(stop - start, device=device)
        if fade > 0:
            left_len = min(fade, stop - start)
            right_len = min(fade, stop - start)
            if left_len > 1:
                local[:left_len] = torch.minimum(
                    local[:left_len],
                    torch.linspace(0.0, 1.0, left_len, device=device),
                )
            if right_len > 1:
                local[-right_len:] = torch.minimum(
                    local[-right_len:],
                    torch.linspace(1.0, 0.0, right_len, device=device),
                )
        gate[start:stop] = torch.maximum(gate[start:stop], local)
    return gate


def oracle_inventory_proposals(record: QCESV5Record) -> list[EventProposal]:
    return [
        EventProposal(
            label=event.label,
            onset_seconds=float(event.onset_seconds),
            offset_seconds=float(event.offset_seconds),
            confidence=1.0,
        )
        for event in record.events
        if getattr(event, "event_kind", "semantic") == "semantic"
    ]


def scene_activity(
    cache: Mapping[str, Any],
    head: ProposalHead | None,
    device: torch.device,
    zeroed: Sequence[str] = (),
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for scene_id, entry in cache["scenes"].items():
        labels, features, rows = features_from_cache(entry)
        features = zero_feature_groups(features, zeroed)
        if head is None:
            activity = energy_activity(rows, entry["mixture"].float())
            onset = None
        else:
            with torch.inference_mode():
                logits, onset_logits, presence = head(features.to(device))
            activity = (
                torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]
            ).cpu()
            onset = torch.sigmoid(onset_logits).cpu()
        result[scene_id] = {
            label: (
                activity[position],
                None if onset is None else onset[position],
            )
            for position, label in enumerate(labels)
        }
    return result


def plan_metrics(
    record: QCESV5Record, plan: EvidencePlan
) -> dict[str, Any]:
    gold_no_evidence = bool(record.no_evidence)
    gold_spans = sorted(
        (
            float(record.event_by_id(event_id).onset_seconds),
            float(record.event_by_id(event_id).offset_seconds),
        )
        for event_id in record.evidence_event_ids
    )
    gold_labels = tuple(
        dict.fromkeys(
            record.event_by_id(event_id).label
            for event_id in record.evidence_event_ids
        )
    )
    return {
        "planned_no_evidence": plan.no_evidence,
        "planned_answer_label": plan.answer_label,
        "planned_labels": list(plan.labels),
        "planner_reason": plan.reason,
        "gold_evidence_labels": list(gold_labels),
        "planner_no_evidence_correct": plan.no_evidence == gold_no_evidence,
        "planner_answer_correct": (
            None
            if gold_no_evidence
            else bool(plan.answer_label == record.answer)
        ),
        "planner_label_set_correct": (
            None
            if gold_no_evidence
            else bool(set(plan.labels) == set(gold_labels))
        ),
        "planner_span_iou_↑": (
            None if gold_no_evidence else interval_iou(plan.spans, gold_spans)
        ),
    }


def interval_iou(
    predicted: Sequence[tuple[float, float]],
    gold: Sequence[tuple[float, float]],
    duration: float = 10.0,
    resolution: int = 2000,
) -> float:
    def rasterise(spans: Sequence[tuple[float, float]]) -> torch.Tensor:
        mask = torch.zeros(resolution)
        for onset, offset in spans:
            start = max(0, min(resolution, int(onset / duration * resolution)))
            stop = max(start, min(resolution, int(offset / duration * resolution)))
            mask[start:stop] = 1.0
        return mask

    left = rasterise(predicted)
    right = rasterise(gold)
    union = float(torch.clamp(left + right, max=1.0).sum())
    return 1.0 if union == 0.0 else float((left * right).sum()) / union


def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = dict(summarize_v5_items(items))
    for key, predicate in (
        ("planner_no_evidence_accuracy_↑", lambda row: row["planner_no_evidence_correct"]),
        ("planner_answer_accuracy_↑", lambda row: row["planner_answer_correct"]),
        ("planner_label_set_accuracy_↑", lambda row: row["planner_label_set_correct"]),
    ):
        values = [predicate(row) for row in items if predicate(row) is not None]
        summary[key] = (
            float(sum(float(value) for value in values) / len(values))
            if values
            else None
        )
    ious = [
        row["planner_span_iou_↑"] for row in items if row["planner_span_iou_↑"] is not None
    ]
    summary["planner_span_iou_mean_↑"] = (
        float(sum(ious) / len(ious)) if ious else None
    )
    # A plan that abstains on an answerable record returns digital silence, and
    # its SD-SDR lands near -70 dB.  Those records are genuine failures, but a
    # bare mean over them says more about the abstention rate than about
    # separation quality, so the abstention rate and a median are reported
    # beside the mean instead of quietly dominating it.
    improvements = [
        float(row["metrics"]["evidence_sd_sdri_db_↑"])
        for row in items
        if row["metrics"]["evidence_sd_sdri_db_↑"] is not None
    ]
    summary["evidence_sd_sdri_answerable_positive_rate_↑"] = (
        float(sum(value > 0.0 for value in improvements) / len(improvements))
        if improvements
        else None
    )
    abstentions = [
        row
        for row in items
        if not row["no_evidence"] and row["planned_no_evidence"]
    ]
    answerable = [row for row in items if not row["no_evidence"]]
    summary["planner_abstention_rate_on_answerable_↓"] = (
        float(len(abstentions) / len(answerable)) if answerable else None
    )
    return summary


def grouped(
    items: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item[field])].append(item)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    manifest = args.manifest.resolve()
    taxonomy = load_taxonomy(args.dataset_config.resolve())
    grid = FrameGrid(sample_rate=32_000)

    dataset = QCESManifestDataset(manifest, crop_samples=None)
    records = [
        record for record in dataset.records if isinstance(record, QCESV5Record)
    ]
    if len(records) != len(dataset.records):
        raise SystemExit("QCES-v6 evaluation requires a pure QCES-v5 manifest")
    # A stem cache may cover only part of a split when the split was capped at
    # caching time.  Restrict the evaluation to the cached scenes and report the
    # resulting coverage, rather than failing or silently mixing two protocols.
    cached_scenes = set(
        torch.load(args.stem_cache, map_location="cpu", weights_only=False)["scenes"]
    )
    manifest_scene_count = len({record.scene_id for record in records})
    manifest_record_count = len(records)
    records = [record for record in records if record.scene_id in cached_scenes]
    if not records:
        raise SystemExit("no manifest record has a cached scene")
    if args.max_records:
        records = records[: args.max_records]

    raw_rows = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            raw_rows[row["id"]] = row

    cache = torch.load(args.stem_cache, map_location="cpu", weights_only=False)
    head: ProposalHead | None = None
    zeroed: tuple[str, ...] = ()
    onset_split = True
    threshold = args.threshold
    if args.proposal_head is not None:
        payload = torch.load(args.proposal_head, map_location="cpu", weights_only=False)
        head = ProposalHead(
            channels=int(payload.get("channels", 96)),
            dropout=float(payload.get("dropout", 0.1)),
        )
        head.load_state_dict(payload["state_dict"])
        head = head.to(device).eval()
        zeroed = tuple(payload.get("zeroed_feature_groups", ()))
        onset_split = bool(payload.get("onset_split", True))
        if threshold is None:
            threshold = float(payload["threshold"])
    if threshold is None:
        threshold = 0.5
    relation_thresholds = parse_relation_thresholds(args.relation_threshold)

    learned_activity = (
        scene_activity(cache, head, device, zeroed) if head is not None else {}
    )
    energy_activity_by_scene = scene_activity(cache, None, device)

    proposal_cache: dict[tuple[str, str, tuple[str, ...], float], list[EventProposal]] = {}

    def proposals_for(
        source: str,
        scene_id: str,
        labels: tuple[str, ...],
        cut: float | None = None,
    ) -> list[EventProposal]:
        if cut is None:
            cut = threshold if source == "learned" else args.energy_threshold
        key = (source, scene_id, labels, float(cut))
        if key in proposal_cache:
            return proposal_cache[key]
        table = learned_activity if source == "learned" else energy_activity_by_scene
        available = [label for label in labels if label in table[scene_id]]
        activity = torch.stack([table[scene_id][label][0] for label in available])
        onsets = [table[scene_id][label][1] for label in available]
        onset_activity = (
            None if onsets[0] is None or not onset_split else torch.stack(onsets)
        )
        decoded = decode_proposals(
            available,
            activity,
            grid,
            threshold=cut,
            onset_activity=onset_activity,
        )
        proposal_cache[key] = decoded
        return decoded

    # Plan every record for every mode before any separator call, so identical
    # prompts across modes and records are encoded and rendered once.
    plans: dict[tuple[str, str], EvidencePlan] = {}
    requests: dict[tuple[str, str], Request | EventwiseRequest] = {}
    for record in records:
        row = raw_rows[record.sample_id]
        parsed = parse_question(row["question"], row["answer_options"], taxonomy)
        labels = tuple(parsed.query_labels)
        for mode in args.modes:
            if mode == "mixture_passthrough":
                plan = EvidencePlan(False, None, (), "mixture_passthrough")
                request = Request("", (), False, False)
            elif mode == "question_prompt__no_gate":
                plan = EvidencePlan(False, None, (), "raw_question_prompt")
                request = Request(row["question"], (), False, False)
            elif mode in {"oracle_text__no_gate", "oracle_text__oracle_gate"}:
                plan = EvidencePlan(
                    bool(record.no_evidence), None, (), "oracle_annotation"
                )
                spans = tuple(
                    (
                        float(record.event_by_id(event_id).onset_seconds),
                        float(record.event_by_id(event_id).offset_seconds),
                    )
                    for event_id in record.evidence_event_ids
                )
                # The oracle uses the same canonical prompt template as the
                # planner and differs only in *which* labels it is given, so
                # the comparison isolates label selection rather than prompt
                # wording, which the frozen separator is very sensitive to.
                oracle_labels = tuple(
                    dict.fromkeys(
                        record.event_by_id(event_id).label
                        for event_id in record.evidence_event_ids
                    )
                )
                request = Request(
                    prompt_for_labels(oracle_labels, describe),
                    spans,
                    mode.endswith("oracle_gate"),
                    bool(record.no_evidence),
                )
            else:
                if mode == "oracle_inventory__planned_gate":
                    inventory = oracle_inventory_proposals(record)
                    inventory = [
                        item for item in inventory if item.label in set(labels)
                    ]
                elif mode.startswith("energy_"):
                    inventory = proposals_for("energy", record.scene_id, labels)
                else:
                    learned_cut = relation_thresholds.get(
                        parsed.relation or "", threshold
                    )
                    inventory = proposals_for(
                        "learned", record.scene_id, labels, learned_cut
                    )
                plan = plan_from_proposals(parsed, inventory)
                if mode in {"eventwise__predicted_gate", "eventwise__predicted_no_gate"}:
                    request = EventwiseRequest(
                        events=tuple(plan.evidence),
                        gated=mode == "eventwise__predicted_gate",
                        silent=plan.no_evidence,
                    )
                elif mode == "predicted__best_effort":
                    # Returning digital silence costs about -80 dB SD-SDR, so a
                    # deployable system reports the no-evidence decision but
                    # still hands back its best guess at the evidence audio.
                    if plan.no_evidence:
                        request = Request(
                            prompt_for_labels(parsed.mentioned_labels, describe),
                            (),
                            False,
                            False,
                        )
                    else:
                        request = Request(
                            prompt_for_labels(plan.labels, describe),
                            plan.spans,
                            True,
                            False,
                        )
                else:
                    request = Request(
                        prompt_for_labels(plan.labels, describe),
                        plan.spans,
                        not mode.endswith("no_gate"),
                        plan.no_evidence,
                    )
            plans[(record.sample_id, mode)] = plan
            requests[(record.sample_id, mode)] = request

    prompts = sorted(
        prompt
        for request in requests.values()
        for prompt in request_prompts(request, describe)
    )
    embeddings = (
        encode_prompts(
            args.audiosep_root.resolve(),
            args.audiosep_checkpoint.resolve(),
            prompts,
            batch_size=args.text_batch_size,
        )
        if prompts
        else {}
    )
    separator = _load_separator(args, device) if prompts else None

    render_modes = set(args.render_mode)
    unknown_render = sorted(render_modes - set(args.modes))
    if unknown_render:
        raise SystemExit("--render-mode not in --modes: " + ", ".join(unknown_render))

    items: list[dict[str, Any]] = []
    separator_calls = 0
    index_by_id = {record.sample_id: index for index, record in enumerate(records)}
    with torch.inference_mode():
        for record in records:
            example = dataset[index_by_id[record.sample_id]]
            mixture = example.mixture.to(device)
            target = example.evidence.to(device)
            target_residual = example.residual.to(device)
            rendered: dict[str, torch.Tensor] = {}
            for mode in args.modes:
                request = requests[(record.sample_id, mode)]
                plan = plans[(record.sample_id, mode)]
                if mode == "mixture_passthrough":
                    evidence = mixture.clone()
                elif request.silent:
                    evidence = torch.zeros_like(mixture)
                elif isinstance(request, EventwiseRequest):
                    evidence = torch.zeros_like(mixture)
                    for label, spans in spans_by_label(request.events).items():
                        prompt = prompt_for_labels((label,), describe)
                        if prompt not in rendered:
                            assert separator is not None
                            condition = embeddings[prompt][None].to(device)
                            rendered[prompt] = separator(
                                {
                                    "mixture": mixture[None, None],
                                    "condition": condition,
                                }
                            )["waveform"][0, 0]
                            separator_calls += 1
                        stem = rendered[prompt]
                        if request.gated:
                            stem = stem * span_gate_with_fade(
                                spans,
                                mixture.numel(),
                                record.sample_rate,
                                device,
                                args.eventwise_padding_seconds,
                                args.eventwise_fade_seconds,
                            )
                        evidence = evidence + stem
                elif not request.prompt:
                    evidence = torch.zeros_like(mixture)
                else:
                    if request.prompt not in rendered:
                        assert separator is not None
                        condition = embeddings[request.prompt][None].to(device)
                        rendered[request.prompt] = separator(
                            {"mixture": mixture[None, None], "condition": condition}
                        )["waveform"][0, 0]
                        separator_calls += 1
                    evidence = rendered[request.prompt]
                    if request.gated:
                        evidence = evidence * span_gate(
                            request.spans,
                            mixture.numel(),
                            record.sample_rate,
                            device,
                            args.gate_dilation_seconds,
                        )
                metrics, descriptives = _v5_item_metrics(
                    no_evidence=bool(record.no_evidence),
                    evidence=evidence,
                    mixture=mixture,
                    target=target,
                    target_residual=target_residual,
                )
                item = {
                    "id": record.sample_id,
                    "mode": mode,
                    "scene_id": record.scene_id,
                    "scene_family_id": record.scene_family_id,
                    "variant_id": record.variant_id,
                    "split": record.split,
                    "evaluation_axis": record.evaluation_axis,
                    "question": record.question,
                    "question_index": record.question_index,
                    "question_type": record.question_type,
                    "relation": record.relation,
                    "answer": record.answer,
                    "no_evidence": bool(record.no_evidence),
                    "prompt": request_prompt_text(request),
                    "predicted_spans": list(request.spans),
                    "metrics": metrics,
                    "descriptives": descriptives,
                    **plan_metrics(record, plan),
                }
                items.append(item)
                if mode in render_modes:
                    write_item_artifacts(
                        question_dir=(
                            output_dir
                            / mode
                            / record.scene_id
                            / f"q{record.question_index}_{record.question_type}"
                        ),
                        item=item,
                        evidence=evidence,
                        residual=mixture - evidence,
                        sample_rate=record.sample_rate,
                        render_audio=True,
                    )

    by_mode = {
        mode: summarize([item for item in items if item["mode"] == mode])
        for mode in args.modes
    }
    by_mode_relation = {
        mode: grouped(
            [item for item in items if item["mode"] == mode], "relation"
        )
        for mode in args.modes
    }
    report = {
        "format": FORMAT_VERSION,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "schema_versions": sorted({record.schema_version for record in records}),
        "stem_cache": str(args.stem_cache.resolve()),
        "proposal_head": (
            str(args.proposal_head.resolve()) if args.proposal_head else None
        ),
        "proposal_threshold": threshold,
        "relation_thresholds": relation_thresholds,
        "zeroed_feature_groups": list(zeroed),
        "onset_split": onset_split,
        "energy_threshold": args.energy_threshold,
        "gate_dilation_seconds": args.gate_dilation_seconds,
        "eventwise_padding_seconds": args.eventwise_padding_seconds,
        "eventwise_fade_seconds": args.eventwise_fade_seconds,
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "audiosep_frozen": True,
        "device": str(device),
        "record_count": len(records),
        "manifest_record_count": manifest_record_count,
        "manifest_scene_count": manifest_scene_count,
        "evaluated_scene_count": len({record.scene_id for record in records}),
        "scene_coverage": len({record.scene_id for record in records})
        / max(manifest_scene_count, 1),
        "separator_calls": separator_calls,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "mode_registry": {mode: MODE_REGISTRY[mode] for mode in args.modes},
        "summaries_by_mode": by_mode,
        "summaries_by_mode_and_relation": by_mode_relation,
        "items": items,
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for mode in args.modes:
        summary = by_mode[mode]
        print(
            f"{mode:38s} SD-SDRi {summary['evidence_sd_sdri_answerable_mean_db_↑']!s:>8.8} "
            f"SI-SDRi {summary['evidence_si_sdri_answerable_mean_db_↑']!s:>8.8} "
            f"ans {summary['planner_answer_accuracy_↑']!s:>6.6} "
            f"noev {summary['planner_no_evidence_accuracy_↑']!s:>6.6} "
            f"IoU {summary['planner_span_iou_mean_↑']!s:>6.6}",
            flush=True,
        )
    print(f"wrote {output_dir / 'evaluation_report.json'}", flush=True)


if __name__ == "__main__":
    main()
