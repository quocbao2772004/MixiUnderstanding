#!/usr/bin/env python3
"""Blinded two-pass annotation UI for the QCES-Real-10 TACOS set.

Run with Streamlit, for example::

    PYTHONPATH=code streamlit run \
      code/mixi_understanding/apps/qces_v5_tacos_annotation_app.py -- \
      --packet data/qces_v5_tacos_real_10s/audit/qces_v5_tacos_annotation_packet.jsonl \
      --audio-receipt data/qces_v5_tacos_real_10s/audit/qces_v5_tacos_audio_receipt.jsonl \
      --response-root outputs/qces_v5_tacos_real_10s_human_annotations

Real-test tasks are hidden unless ``--allow-real-test`` is explicitly passed.
The app never loads model outputs.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_tacos import (  # noqa: E402
    TacosAuditError,
)
from mixi_understanding.data.qces_v5_tacos_annotation import (  # noqa: E402
    RESPONSE_FORMAT,
    load_bound_tasks,
    response_path,
    save_response,
    validate_response,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--packet",
        type=Path,
        default=PROJECT_ROOT
        / "data/qces_v5_tacos_real_10s/audit/qces_v5_tacos_annotation_packet.jsonl",
    )
    parser.add_argument(
        "--audio-receipt",
        type=Path,
        default=PROJECT_ROOT
        / "data/qces_v5_tacos_real_10s/audit/qces_v5_tacos_audio_receipt.jsonl",
    )
    parser.add_argument(
        "--response-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_v5_tacos_real_10s_human_annotations",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--allow-real-test", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def _existing_response(
    response_root: Path,
    rater_id: str,
    task: Mapping[str, Any],
    packet_fingerprint: str,
    audio_receipt_fingerprint: str,
) -> Mapping[str, Any] | None:
    try:
        path = response_path(response_root, rater_id, task["scene_id"])
    except TacosAuditError:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError(f"cannot read saved response: {path}") from error
    if not isinstance(payload, dict):
        raise TacosAuditError(f"saved response is not an object: {path}")
    try:
        return validate_response(
            payload,
            task=task,
            packet_fingerprint=packet_fingerprint,
            audio_receipt_fingerprint=audio_receipt_fingerprint,
        )
    except TacosAuditError as error:
        raise TacosAuditError(
            f"saved response is stale/invalid: {path}: {error}"
        ) from error


def _default_region(
    region_id: str,
    proposal: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if existing is not None:
        for row in existing.get("regions", []):
            if row.get("region_id") == region_id:
                return row
    return {
        "region_id": region_id,
        "audible": "uncertain",
        "proposal_caption_accurate": "uncertain",
        "canonical_event_phrase": "",
        "verified_onset_seconds": float(proposal["onset_seconds"]),
        "verified_offset_seconds": float(proposal["offset_seconds"]),
        "contamination_rating": "uncertain",
        "salience_1_to_5": 3,
    }


def main() -> None:
    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="QCES-Real-10 annotation", layout="wide")
    st.title("QCES-Real-10 · blinded TACOS event annotation")
    st.caption(
        "Only the immutable derived 10 s WAV, crop-relative TACOS proposals, and "
        "construction metadata are shown. No original full clip or model output is loaded."
    )
    try:
        packet_fingerprint, audio_fingerprint, tasks = load_bound_tasks(
            packet_path=args.packet.resolve(),
            audio_receipt_path=args.audio_receipt.resolve(),
            project_root=args.project_root.resolve(),
        )
    except (OSError, TacosAuditError) as error:
        st.error(str(error))
        st.stop()

    allowed_tasks = [
        task
        for task in tasks
        if args.allow_real_test or task["selection_partition"] != "real_test"
    ]
    if not allowed_tasks:
        st.error("No tasks are visible under the current split policy.")
        st.stop()
    if args.allow_real_test:
        st.warning(
            "REAL TEST IS VISIBLE. Use this mode only for independent dataset "
            "annotators; do not inspect model outputs or tune the method from these clips."
        )

    rater_id = st.sidebar.text_input(
        "Rater ID",
        value=st.session_state.get("qces_tacos_rater_id", ""),
        help="Use a stable pseudonymous ID, e.g. rater_a01.",
    ).strip()
    st.session_state["qces_tacos_rater_id"] = rater_id
    scopes = sorted(
        {
            f"{task['selection_partition']}:{task['selection_tier']}"
            for task in allowed_tasks
        }
    )
    scope = st.sidebar.selectbox("Task scope", scopes, index=0)
    scoped = [
        task
        for task in allowed_tasks
        if f"{task['selection_partition']}:{task['selection_tier']}" == scope
    ]
    scoped.sort(key=lambda task: (task["selection_ordinal"], task["scene_id"]))

    completed = 0
    if rater_id:
        try:
            completed = sum(
                _existing_response(
                    args.response_root,
                    rater_id,
                    candidate_task,
                    packet_fingerprint,
                    audio_fingerprint,
                )
                is not None
                for candidate_task in scoped
            )
        except TacosAuditError as error:
            st.error(str(error))
            st.stop()
    st.sidebar.metric("Saved in this scope ↑", f"{completed}/{len(scoped)}")
    labels = [
        f"{index + 1:03d}/{len(scoped):03d} · {task['scene_id']} · "
        f"{task['source']['subclass']}"
        for index, task in enumerate(scoped)
    ]
    selected_label = st.sidebar.selectbox("Scene", labels, index=0)
    task = scoped[labels.index(selected_label)]
    try:
        existing = (
            _existing_response(
                args.response_root,
                rater_id,
                task,
                packet_fingerprint,
                audio_fingerprint,
            )
            if rater_id
            else None
        )
    except TacosAuditError as error:
        st.error(str(error))
        st.stop()

    st.subheader(
        f"{task['scene_id']} · {task['selection_partition']} / "
        f"{task['selection_tier']}"
    )
    source_columns = st.columns(5)
    source_columns[0].metric(
        "Final WAV duration", f"{task['annotation_audio']['duration_seconds']:.2f}s"
    )
    source_columns[1].metric("Frames", f"{task['annotation_audio']['num_frames']:,}")
    source_columns[2].metric(
        "Usable proposals ↑", task["proposal_diagnostics"]["usable_region_count_↑"]
    )
    source_columns[3].metric(
        "Overlap pairs ↑", task["proposal_diagnostics"]["overlap_pair_count_↑"]
    )
    source_columns[4].metric("Source license", task["source"]["audio_license_spdx"])
    upstream_window = task["source"][
        "benchmark_window_interval_in_upstream_clip_seconds"
    ]
    st.caption(
        "Frozen crop metadata: "
        f"upstream-clip {float(upstream_window[0]):.6f}–"
        f"{float(upstream_window[1]):.6f}s · "
        f"mono {task['annotation_audio']['sample_rate_hz']} Hz · "
        f"WAV SHA-256 {task['annotation_audio']['local_sha256'][:16]}…. "
        "The original variable-duration source is intentionally unavailable here."
    )
    st.audio(task["resolved_audio_path"], format="audio/wav")

    eligible = {
        row["region_id"]: row
        for row in task["proposal_regions"]
        if row["eligible_proposal"]
    }
    timeline_rows = [
        {
            "region_id": row["region_id"],
            "onset_s": round(float(row["onset_seconds"]), 3),
            "offset_s": round(float(row["offset_seconds"]), 3),
            "caption_proposal": row["caption"],
            "suggested_chain": row["region_id"]
            in task["suggested_distinct_onset_chain"][:4],
        }
        for row in task["proposal_regions"]
    ]
    st.dataframe(timeline_rows, use_container_width=True, hide_index=True)
    default_selected = (
        existing.get("selected_region_ids", [])
        if existing is not None
        else task["suggested_distinct_onset_chain"][:4]
    )
    chosen = st.multiselect(
        "Choose exactly four distinct audible events in onset order",
        options=list(eligible),
        default=[value for value in default_selected if value in eligible],
        format_func=lambda region_id: (
            f"{region_id} · {eligible[region_id]['onset_seconds']:.2f}–"
            f"{eligible[region_id]['offset_seconds']:.2f}s · "
            f"{eligible[region_id]['caption']}"
        ),
        max_selections=4,
    )
    chosen = sorted(chosen, key=lambda value: eligible[value]["onset_seconds"])
    if len(chosen) != 4:
        st.info("Select exactly four regions before saving this scene.")

    with st.form(f"annotation-{task['scene_id']}"):
        region_results: list[dict[str, Any]] = []
        for rank, region_id in enumerate(chosen):
            proposal = eligible[region_id]
            defaults = _default_region(region_id, proposal, existing)
            st.markdown(
                f"**Verified event rank {rank} — {region_id}**  "
                f"(proposal: {proposal['caption']})"
            )
            columns = st.columns([1, 1.3, 2, 1, 1, 1, 1])
            audible = columns[0].selectbox(
                "Audible?",
                ["yes", "no", "uncertain"],
                index=["yes", "no", "uncertain"].index(defaults["audible"]),
                key=f"{task['scene_id']}-{region_id}-audible",
            )
            proposal_caption_accurate = columns[1].selectbox(
                "Proposal accurate?",
                ["yes", "no", "uncertain"],
                index=["yes", "no", "uncertain"].index(
                    defaults["proposal_caption_accurate"]
                ),
                key=f"{task['scene_id']}-{region_id}-proposal-accurate",
                help=(
                    "Does the immutable TACOS proposal caption shown above "
                    "accurately describe the audible event? Accepted scenes require yes."
                ),
            )
            phrase = columns[2].text_input(
                "Concise event phrase",
                value=defaults["canonical_event_phrase"],
                key=f"{task['scene_id']}-{region_id}-phrase",
            )
            onset = columns[3].number_input(
                "Onset (s)",
                min_value=0.0,
                max_value=10.0,
                value=float(defaults["verified_onset_seconds"]),
                step=0.01,
                format="%.3f",
                key=f"{task['scene_id']}-{region_id}-onset",
            )
            offset = columns[4].number_input(
                "Offset (s)",
                min_value=0.0,
                max_value=10.0,
                value=float(defaults["verified_offset_seconds"]),
                step=0.01,
                format="%.3f",
                key=f"{task['scene_id']}-{region_id}-offset",
            )
            contamination = columns[5].selectbox(
                "Region",
                ["clean", "mixed", "uncertain"],
                index=["clean", "mixed", "uncertain"].index(
                    defaults["contamination_rating"]
                ),
                key=f"{task['scene_id']}-{region_id}-contamination",
            )
            salience = columns[6].slider(
                "Salience",
                min_value=1,
                max_value=5,
                value=int(defaults["salience_1_to_5"]),
                key=f"{task['scene_id']}-{region_id}-salience",
            )
            region_results.append(
                {
                    "region_id": region_id,
                    "audible": audible,
                    "proposal_caption_accurate": proposal_caption_accurate,
                    "canonical_event_phrase": phrase,
                    "verified_onset_seconds": onset,
                    "verified_offset_seconds": offset,
                    "contamination_rating": contamination,
                    "salience_1_to_5": salience,
                }
            )

        default_inventory = (
            existing.get("event_inventory_complete_for_relations", "uncertain")
            if existing
            else "uncertain"
        )
        inventory = st.selectbox(
            "Is the audible event inventory complete enough to judge immediate onset order?",
            ["yes", "no", "uncertain"],
            index=["yes", "no", "uncertain"].index(default_inventory),
        )
        existing_adjacency = (
            existing.get("adjacent_onset_relations_verified", ["uncertain"] * 3)
            if existing
            else ["uncertain"] * 3
        )
        adjacency: list[str] = []
        adjacency_columns = st.columns(3)
        for index, column in enumerate(adjacency_columns):
            adjacency.append(
                column.selectbox(
                    f"Rank {index} → {index + 1} is immediate?",
                    ["yes", "no", "uncertain"],
                    index=["yes", "no", "uncertain"].index(existing_adjacency[index]),
                    key=f"{task['scene_id']}-adjacency-{index}",
                )
            )

        st.markdown("**Candidate absent-anchor audit**")
        st.caption(
            "Judge every immutable caption against the entire final WAV. The "
            "finalizer will later choose only a jointly inaudible caption whose "
            "positive use remains in the released split."
        )
        previous_absent_judgments = {
            row.get("caption"): row.get("confirmed_inaudible", "uncertain")
            for row in (existing.get("absent_anchor_judgments", []) if existing else [])
            if isinstance(row, Mapping)
        }
        absent_judgments: list[dict[str, str]] = []
        absent_columns = st.columns(2)
        for index, candidate in enumerate(
            task["question_contract"]["absent_anchor_candidates"]
        ):
            previous_verdict = previous_absent_judgments.get(candidate, "uncertain")
            verdict = absent_columns[index % 2].selectbox(
                f"Inaudible throughout? · {candidate}",
                ["yes", "no", "uncertain"],
                index=["yes", "no", "uncertain"].index(previous_verdict),
                key=f"{task['scene_id']}-absent-{index}",
            )
            absent_judgments.append(
                {"caption": candidate, "confirmed_inaudible": verdict}
            )
        previous_listen = existing.get("full_listen_confirmation") if existing else None
        listen_is_bound = bool(
            isinstance(previous_listen, Mapping)
            and previous_listen.get("entire_final_wav_listened") is True
            and previous_listen.get("file_sha256")
            == task["annotation_audio"]["local_sha256"]
            and previous_listen.get("pcm_f32le_sha256")
            == task["annotation_audio"]["pcm_f32le_sha256"]
        )
        entire_wav_listened = st.checkbox(
            "I listened to this exact final WAV from 0.000 s through 10.000 s.",
            value=listen_is_bound,
            help=(
                "Required for every accept/reject/uncertain response. The confirmation "
                "is cryptographically bound to the displayed WAV and decoded PCM."
            ),
        )
        decision = st.selectbox(
            "Scene decision",
            ["accept", "reject", "uncertain"],
            index=["accept", "reject", "uncertain"].index(
                existing.get("scene_decision", "uncertain") if existing else "uncertain"
            ),
        )
        notes = st.text_area(
            "Notes",
            value=existing.get("notes", "") if existing else "",
        )
        submitted = st.form_submit_button("Save blinded annotation")

    if submitted:
        if not rater_id:
            st.error("Enter a valid rater ID first.")
        else:
            response = {
                "format": RESPONSE_FORMAT,
                "packet_fingerprint": packet_fingerprint,
                "audio_receipt_fingerprint": audio_fingerprint,
                "scene_id": task["scene_id"],
                "rater_id": rater_id,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "scene_decision": decision,
                "selected_region_ids": chosen,
                "regions": region_results,
                "event_inventory_complete_for_relations": inventory,
                "adjacent_onset_relations_verified": adjacency,
                "absent_anchor_judgments": absent_judgments,
                "full_listen_confirmation": {
                    "entire_final_wav_listened": entire_wav_listened,
                    "file_sha256": task["annotation_audio"]["local_sha256"],
                    "pcm_f32le_sha256": task["annotation_audio"]["pcm_f32le_sha256"],
                    "sample_rate_hz": task["annotation_audio"]["sample_rate_hz"],
                    "num_frames": task["annotation_audio"]["num_frames"],
                    "duration_seconds": task["annotation_audio"]["duration_seconds"],
                },
                "model_outputs_visible": False,
                "notes": notes,
            }
            try:
                path = save_response(
                    response_root=args.response_root,
                    response=response,
                    task=task,
                    packet_fingerprint=packet_fingerprint,
                    audio_receipt_fingerprint=audio_fingerprint,
                )
            except (OSError, TacosAuditError) as error:
                st.error(str(error))
            else:
                st.success(f"Saved: {path}")


if __name__ == "__main__":
    main()
