#!/usr/bin/env python3
"""Prototype multi-speaker diarization plus local overlap separation.

This does not pretend that a two-output separator can separate an arbitrary
number of speakers over a full recording.  It first estimates speaker turns,
then calls the existing two-speaker separator only around adjacent turn
boundaries and stitches the locally separated sources into one full-length
stem per estimated speaker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from mixi_understanding.scripts.serve_qces_live_inference import SpeakerDiarizer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RATE = 16_000
LIVE_URL = "http://127.0.0.1:8512/infer"
SEPARATOR_URL = "http://127.0.0.1:8517/separate"
MODEL_NAME = "MossFormer2_SS_16K"


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("error"):
        raise RuntimeError(f"{result['error']}: {result.get('message', '')}")
    return result


def _read_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(audio.mean(axis=1), dtype=np.float32)
    if rate != SAMPLE_RATE:
        divisor = int(np.gcd(rate, SAMPLE_RATE))
        mono = resample_poly(
            mono,
            SAMPLE_RATE // divisor,
            rate // divisor,
        ).astype(np.float32)
    return mono


def _fit_length(audio: np.ndarray, samples: int) -> np.ndarray:
    result = np.zeros(samples, dtype=np.float32)
    result[: min(samples, len(audio))] = audio[:samples]
    return result


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def _normalise(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _speaker_region(prediction: dict[str, Any], duration: float) -> tuple[float, float]:
    span = prediction.get("speech_span_seconds")
    if isinstance(span, list) and len(span) == 2:
        return max(0.0, float(span[0])), min(duration, float(span[1]))
    speech = [
        event for event in prediction.get("predicted_events", [])
        if event.get("label") == "Speech"
    ]
    if speech:
        return (
            max(0.0, min(float(row["start_seconds"]) for row in speech)),
            min(duration, max(float(row["end_seconds"]) for row in speech)),
        )
    return 0.0, duration


def _event_spans(prediction: dict[str, Any]) -> list[tuple[float, float, str]]:
    ignored = {"Speech", "Conversation"}
    return [
        (
            float(row["start_seconds"]),
            float(row["end_seconds"]),
            str(row["label"]),
        )
        for row in prediction.get("predicted_events", [])
        if row.get("label") not in ignored
    ]


def _window_audio(
    waveform: np.ndarray,
    starts: list[float],
    window_seconds: float,
) -> list[np.ndarray]:
    samples = int(round(window_seconds * SAMPLE_RATE))
    values: list[np.ndarray] = []
    for start in starts:
        left = int(round(start * SAMPLE_RATE))
        values.append(_fit_length(waveform[left : left + samples], samples))
    return values


def _select_speaker_count(
    embeddings: np.ndarray,
    keep: np.ndarray,
    max_speakers: int,
    minimum_cluster_windows: int,
    complexity_tolerance: float = 0.02,
) -> tuple[int, np.ndarray, list[dict[str, Any]]]:
    eligible = embeddings[keep]
    audits: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, np.ndarray]] = []
    maximum = min(max_speakers, len(eligible) // minimum_cluster_windows)
    for count in range(2, maximum + 1):
        labels = AgglomerativeClustering(
            n_clusters=count,
            metric="cosine",
            linkage="average",
        ).fit_predict(eligible)
        counts = np.bincount(labels, minlength=count)
        score = float(silhouette_score(eligible, labels, metric="cosine"))
        valid = int(counts.min()) >= minimum_cluster_windows
        audits.append(
            {
                "speaker_count": count,
                "silhouette": score,
                "cluster_window_counts": counts.tolist(),
                "valid": valid,
            }
        )
        if valid:
            candidates.append((score, count, labels))
    if not candidates:
        return 1, np.zeros(int(keep.sum()), dtype=np.int64), audits
    best_score = max(row[0] for row in candidates)
    # Prefer the smallest count statistically indistinguishable from the best.
    # This prevents a tiny silhouette gain from inventing another speaker.
    near_best = [
        row for row in candidates
        if row[0] >= best_score - max(0.0, float(complexity_tolerance))
    ]
    _, selected_count, selected_labels = min(near_best, key=lambda row: row[1])
    if best_score < 0.25:
        return 1, np.zeros(int(keep.sum()), dtype=np.int64), audits
    return selected_count, selected_labels.astype(np.int64), audits


def _decode_turns(
    embeddings: np.ndarray,
    starts: list[float],
    keep: np.ndarray,
    eligible_labels: np.ndarray,
    speech_start: float,
    speech_end: float,
    window_seconds: float,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], np.ndarray]:
    labels = np.full(len(starts), -1, dtype=np.int64)
    labels[keep] = eligible_labels
    kept_indices = np.flatnonzero(keep)
    if kept_indices.size == 0:
        labels[:] = 0
    else:
        first = int(kept_indices[0])
        labels[:first] = labels[first]
        # A detected environmental event can cover most of a speaker turn.
        # Forward filling preserves the last confirmed identity instead of
        # treating the event itself as a new speaker cluster.
        for index in range(first + 1, len(labels)):
            if labels[index] < 0:
                labels[index] = labels[index - 1]
    labels = median_filter(labels, size=3, mode="nearest").astype(np.int64)

    first_seen: dict[int, float] = {}
    for label, start in zip(labels.tolist(), starts, strict=True):
        first_seen.setdefault(int(label), float(start))
    ordered = sorted(first_seen, key=first_seen.get)
    speaker_index = {label: index for index, label in enumerate(ordered)}
    speaker_labels = np.asarray([speaker_index[int(label)] for label in labels], dtype=np.int64)

    centers = np.asarray(starts, dtype=np.float64) + window_seconds / 2.0
    boundaries = np.concatenate(
        (
            np.asarray([speech_start]),
            (centers[:-1] + centers[1:]) / 2.0,
            np.asarray([speech_end]),
        )
    )
    turns: list[dict[str, Any]] = []
    for index, label in enumerate(speaker_labels.tolist()):
        start = float(boundaries[index])
        end = float(boundaries[index + 1])
        if (
            turns
            and int(turns[-1]["speaker_index"]) == label
            and start - float(turns[-1]["end_seconds"]) <= 0.03
        ):
            turns[-1]["end_seconds"] = end
        else:
            turns.append(
                {
                    "speaker_index": label,
                    "speaker_id": f"SPEAKER_{label + 1:02d}",
                    "start_seconds": start,
                    "end_seconds": end,
                }
            )

    centroids: dict[int, np.ndarray] = {}
    for cluster_label, index in speaker_index.items():
        rows = embeddings[keep & (labels == cluster_label)]
        if not len(rows):
            rows = embeddings[labels == cluster_label]
        centroids[index] = _normalise(np.mean(rows, axis=0))
    return turns, centroids, speaker_labels


def _fade_mask(start: int, end: int, samples: int, fade_seconds: float = 0.025) -> np.ndarray:
    mask = np.zeros(samples, dtype=np.float32)
    start = max(0, min(samples, start))
    end = max(start, min(samples, end))
    mask[start:end] = 1.0
    fade = min(int(round(fade_seconds * SAMPLE_RATE)), (end - start) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mask[start : start + fade] *= ramp
        mask[end - fade : end] *= ramp[::-1]
    return mask


def _raw_separator_outputs(receipt: dict[str, Any]) -> list[Path]:
    listening = [Path(row["audio_path"]) for row in receipt.get("speakers", [])]
    if len(listening) != 2:
        raise RuntimeError("local separator did not return two sources")
    run_dir = listening[0].parent
    generated = sorted(
        path
        for path in (run_dir / MODEL_NAME).glob("*_s[12].*")
        if path.suffix.lower() in {".wav", ".flac"}
    )
    return generated if len(generated) == 2 else listening


def _local_separation(
    input_path: Path,
    output_dir: Path,
    transition_index: int,
    waveform: np.ndarray,
    speech_view: np.ndarray,
    start_seconds: float,
    end_seconds: float,
    adjacent_speakers: tuple[int, int],
    centroids: dict[int, np.ndarray],
    diarizer: SpeakerDiarizer,
    minimum_directional_consistency: float,
    minimum_assignment_margin: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any], bool]:
    left = max(0, int(round(start_seconds * SAMPLE_RATE)))
    right = min(len(waveform), int(round(end_seconds * SAMPLE_RATE)))
    region_path = output_dir / "regions" / f"transition_{transition_index:02d}.wav"
    _write(region_path, waveform[left:right])
    response = _post_json(
        SEPARATOR_URL,
        {"audio_path": str(region_path.resolve())},
        timeout=300,
    )
    source_paths = _raw_separator_outputs(response)
    sources = [_fit_length(_read_mono(path), right - left) for path in source_paths]
    source_embeddings = diarizer._embeddings(sources)
    target_embeddings = np.stack([centroids[index] for index in adjacent_speakers])
    similarities = source_embeddings @ target_embeddings.T
    source_rows, target_columns = linear_sum_assignment(-similarities)
    assignment = {
        adjacent_speakers[int(target_column)]: sources[int(source_row)]
        for source_row, target_column in zip(source_rows, target_columns, strict=True)
    }

    # A two-source model emits two outputs even at an ordinary, non-overlapped
    # speaker change.  Apply it only when both outputs have a clear identity and
    # their energy follows the expected old-speaker/new-speaker direction.
    source_for_speaker = {
        adjacent_speakers[int(target_column)]: int(source_row)
        for source_row, target_column in zip(source_rows, target_columns, strict=True)
    }
    midpoint = len(sources[0]) // 2
    direction_scores: dict[int, float] = {}
    for speaker, source_row in source_for_speaker.items():
        energy = np.square(sources[source_row].astype(np.float64))
        before = float(np.sum(energy[:midpoint]))
        after = float(np.sum(energy[midpoint:]))
        fraction_before = before / max(before + after, 1e-12)
        direction_scores[speaker] = (
            fraction_before if speaker == adjacent_speakers[0] else 1.0 - fraction_before
        )
    directional_consistency = float(np.mean(list(direction_scores.values())))
    assignment_margins: list[float] = []
    for source_row, target_column in zip(source_rows, target_columns, strict=True):
        selected = float(similarities[source_row, target_column])
        alternative = float(similarities[source_row, 1 - target_column])
        assignment_margins.append(selected - alternative)
    mean_assignment_margin = float(np.mean(assignment_margins))
    should_apply = (
        directional_consistency >= minimum_directional_consistency
        and mean_assignment_margin >= minimum_assignment_margin
    )

    combined = np.sum(np.stack(list(assignment.values())), axis=0)
    target = speech_view[left:right]
    scale = float(np.dot(combined, target) / (np.dot(combined, combined) + 1e-12))
    scale = float(np.clip(scale, 0.20, 5.0))
    assignment = {key: value * scale for key, value in assignment.items()}
    audit = {
        "transition_index": transition_index,
        "start_seconds": left / SAMPLE_RATE,
        "end_seconds": right / SAMPLE_RATE,
        "adjacent_speakers": [f"SPEAKER_{index + 1:02d}" for index in adjacent_speakers],
        "source_paths": [str(path.resolve()) for path in source_paths],
        "cosine_similarity": similarities.tolist(),
        "source_to_speaker": {
            str(int(source_row) + 1): f"SPEAKER_{adjacent_speakers[int(target_column)] + 1:02d}"
            for source_row, target_column in zip(source_rows, target_columns, strict=True)
        },
        "joint_amplitude_scale": scale,
        "directional_consistency": directional_consistency,
        "minimum_directional_consistency": minimum_directional_consistency,
        "mean_assignment_margin": mean_assignment_margin,
        "minimum_assignment_margin": minimum_assignment_margin,
        "applied": should_apply,
        "decision": "local_separation" if should_apply else "diarization_only",
    }
    return assignment, audit, should_apply


def _sisdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = estimate.astype(np.float64) - float(np.mean(estimate))
    reference = reference.astype(np.float64) - float(np.mean(reference))
    projection = reference * (
        float(np.dot(estimate, reference)) / (float(np.dot(reference, reference)) + 1e-12)
    )
    noise = estimate - projection
    return float(10.0 * np.log10((np.sum(projection**2) + 1e-12) / (np.sum(noise**2) + 1e-12)))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _wer(hypothesis: str, reference: str) -> float:
    hyp, ref = _tokens(hypothesis), _tokens(reference)
    previous = list(range(len(hyp) + 1))
    for row, expected in enumerate(ref, start=1):
        current = [row]
        for column, predicted in enumerate(hyp, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(expected != predicted),
                )
            )
        previous = current
    return previous[-1] / max(1, len(ref))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--speech-view", type=Path, default=None)
    parser.add_argument("--prediction", type=Path, default=None)
    parser.add_argument("--oracle-dir", type=Path, default=None)
    parser.add_argument("--scene", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_multispeaker_region_separator_v1",
    )
    parser.add_argument("--max-speakers", type=int, default=4)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.20)
    parser.add_argument("--event-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--transition-radius-seconds", type=float, default=0.65)
    parser.add_argument("--minimum-directional-consistency", type=float, default=0.75)
    parser.add_argument("--minimum-assignment-margin", type=float, default=0.40)
    parser.add_argument("--skip-local-separation", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    waveform = _read_mono(input_path)
    duration = len(waveform) / SAMPLE_RATE
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.prediction:
        prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    else:
        prediction = _post_json(
            LIVE_URL,
            {"audio_path": str(input_path), "verify_speakers": False},
            timeout=180,
        )
    speech_view = (
        _fit_length(_read_mono(args.speech_view.resolve()), len(waveform))
        if args.speech_view
        else waveform.copy()
    )
    speech_start, speech_end = _speaker_region(prediction, duration)
    event_spans = _event_spans(prediction)

    latest = max(speech_start, speech_end - args.window_seconds)
    starts = list(np.arange(speech_start, latest + 1e-7, args.hop_seconds))
    if not starts or latest - starts[-1] > 0.04:
        starts.append(latest)
    starts = sorted(set(round(float(value), 6) for value in starts))
    windows = _window_audio(waveform, starts, args.window_seconds)
    diarizer = SpeakerDiarizer("cpu")
    embeddings = diarizer._embeddings(windows)
    keep = np.asarray(
        [
            max(
                [
                    _overlap(start, start + args.window_seconds, event_start, event_end)
                    for event_start, event_end, _ in event_spans
                ]
                or [0.0]
            )
            / args.window_seconds
            < args.event_overlap_ratio
            for start in starts
        ],
        dtype=bool,
    )
    if int(keep.sum()) < 4:
        keep[:] = True

    speaker_count, eligible_labels, count_audit = _select_speaker_count(
        embeddings,
        keep,
        max(1, args.max_speakers),
        minimum_cluster_windows=2,
    )
    turns, centroids, _ = _decode_turns(
        embeddings,
        starts,
        keep,
        eligible_labels,
        speech_start,
        speech_end,
        args.window_seconds,
    )
    speaker_count = len({int(row["speaker_index"]) for row in turns})

    stems = [np.zeros_like(waveform) for _ in range(speaker_count)]
    for turn in turns:
        left = int(round(float(turn["start_seconds"]) * SAMPLE_RATE))
        right = int(round(float(turn["end_seconds"]) * SAMPLE_RATE))
        mask = _fade_mask(left, right, len(waveform))
        stems[int(turn["speaker_index"])] += speech_view * mask

    transition_audits: list[dict[str, Any]] = []
    if not args.skip_local_separation:
        for transition_index, (previous, current) in enumerate(zip(turns, turns[1:]), start=1):
            old_speaker = int(previous["speaker_index"])
            new_speaker = int(current["speaker_index"])
            if old_speaker == new_speaker:
                continue
            boundary = (
                float(previous["end_seconds"]) + float(current["start_seconds"])
            ) / 2.0
            start = max(speech_start, boundary - args.transition_radius_seconds)
            end = min(speech_end, boundary + args.transition_radius_seconds)
            assignment, audit, should_apply = _local_separation(
                input_path,
                output_dir,
                transition_index,
                waveform,
                speech_view,
                start,
                end,
                (old_speaker, new_speaker),
                centroids,
                diarizer,
                args.minimum_directional_consistency,
                args.minimum_assignment_margin,
            )
            if should_apply:
                left = int(round(start * SAMPLE_RATE))
                right = min(len(waveform), left + len(next(iter(assignment.values()))))
                blend = np.sin(
                    np.linspace(0.0, np.pi, right - left, dtype=np.float32)
                ) ** 2
                for speaker in (old_speaker, new_speaker):
                    local = assignment[speaker][: right - left]
                    stems[speaker][left:right] = (
                        stems[speaker][left:right] * (1.0 - blend) + local * blend
                    )
            transition_audits.append(audit)

    speaker_rows: list[dict[str, Any]] = []
    references: list[str] = []
    if args.scene:
        scene = json.loads(args.scene.read_text(encoding="utf-8"))
        references = [str(row.get("text") or "") for row in scene.get("speakers", [])]
    for speaker_index, stem in enumerate(stems):
        path = output_dir / f"speaker_{speaker_index + 1:02d}.wav"
        peak = float(np.max(np.abs(stem)))
        if peak > 0.95:
            stem = stem * (0.95 / peak)
            stems[speaker_index] = stem
        _write(path, stem)
        row: dict[str, Any] = {
            "speaker_id": f"SPEAKER_{speaker_index + 1:02d}",
            "audio_path": str(path.resolve()),
        }
        if not args.skip_asr:
            asr = _post_json(
                LIVE_URL,
                {"audio_path": str(path.resolve()), "verify_speakers": False},
                timeout=180,
            )
            row["hypothesis"] = str(asr.get("hypothesis") or "").strip()
            if speaker_index < len(references):
                row["reference"] = references[speaker_index]
                row["wer"] = _wer(row["hypothesis"], references[speaker_index])
        speaker_rows.append(row)

    metrics: dict[str, Any] = {}
    if args.oracle_dir:
        oracle_paths = sorted(args.oracle_dir.resolve().glob("speaker_*.wav"))
        if len(oracle_paths) >= speaker_count:
            source_metrics = []
            for index, estimate in enumerate(stems):
                reference = _fit_length(_read_mono(oracle_paths[index]), len(estimate))
                predicted_sisdr = _sisdr(estimate, reference)
                mixture_sisdr = _sisdr(waveform, reference)
                source_metrics.append(
                    {
                        "speaker_id": f"SPEAKER_{index + 1:02d}",
                        "si_sdr_db": predicted_sisdr,
                        "mixture_si_sdr_db": mixture_sisdr,
                        "si_sdri_db": predicted_sisdr - mixture_sisdr,
                    }
                )
            metrics["oracle_source_metrics"] = source_metrics
            metrics["mean_si_sdr_db"] = float(
                np.mean([row["si_sdr_db"] for row in source_metrics])
            )
            metrics["mean_si_sdri_db"] = float(
                np.mean([row["si_sdri_db"] for row in source_metrics])
            )
    wers = [float(row["wer"]) for row in speaker_rows if "wer" in row]
    if wers:
        metrics["mean_wer"] = float(np.mean(wers))

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    receipt = {
        "format": "qces_multispeaker_region_separator_v2",
        "complete": True,
        "input_path": str(input_path),
        "input_sha256": digest,
        "annotation_free_inference": True,
        "speaker_count": speaker_count,
        "speech_region_seconds": [speech_start, speech_end],
        "event_masked_spans": [
            {"start_seconds": start, "end_seconds": end, "label": label}
            for start, end, label in event_spans
        ],
        "diarization": {
            "window_seconds": args.window_seconds,
            "hop_seconds": args.hop_seconds,
            "kept_windows": int(keep.sum()),
            "total_windows": len(starts),
            "count_candidates": count_audit,
            "selected_speaker_count": speaker_count,
            "turns": turns,
        },
        "transition_separation": transition_audits,
        "speakers": speaker_rows,
        "metrics": metrics,
        "limitations": [
            "Speaker count is inferred from a short recording and is not guaranteed.",
            "The local separator supports at most two simultaneous speakers per transition region.",
            "Evaluation metrics are only present when synthetic oracle stems are supplied.",
        ],
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
