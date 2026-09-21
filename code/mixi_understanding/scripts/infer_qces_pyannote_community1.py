#!/usr/bin/env python3
"""Run Community-1 diarization and optionally render two-speaker evidence.

Community-1 is used as an overlap-aware gate.  Waveform separation is only
requested when diarization finds exactly two speakers and at least one region
where both speakers are simultaneously active.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


MODEL_ID = "pyannote/speaker-diarization-community-1"
SEPARATOR_URL = "http://127.0.0.1:8517/separate"


def _post_json(url: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
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


def _tracks(annotation: Any) -> list[dict[str, Any]]:
    return [
        {
            "start_seconds": float(segment.start),
            "end_seconds": float(segment.end),
            "speaker_id": str(speaker),
        }
        for segment, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _overlap_regions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries = sorted(
        {
            float(value)
            for row in rows
            for value in (row["start_seconds"], row["end_seconds"])
        }
    )
    regions: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = sorted(
            {
                str(row["speaker_id"])
                for row in rows
                if float(row["start_seconds"]) < end
                and float(row["end_seconds"]) > start
            }
        )
        if len(active) < 2:
            continue
        if (
            regions
            and regions[-1]["speaker_ids"] == active
            and abs(float(regions[-1]["end_seconds"]) - start) <= 1e-6
        ):
            regions[-1]["end_seconds"] = end
        else:
            regions.append(
                {
                    "start_seconds": start,
                    "end_seconds": end,
                    "speaker_ids": active,
                }
            )
    return regions


def _embedding_audit(output: Any) -> dict[str, Any]:
    embeddings = np.asarray(output.speaker_embeddings, dtype=np.float32)
    if not embeddings.size:
        return {"shape": list(embeddings.shape), "valid_rows": [], "cosine_similarity": []}
    norms = np.linalg.norm(embeddings, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-8)
    normalized = np.zeros_like(embeddings)
    normalized[valid] = embeddings[valid] / norms[valid, None]
    return {
        "shape": list(embeddings.shape),
        "valid_rows": valid.tolist(),
        "cosine_similarity": (normalized @ normalized.T).tolist(),
    }


def _render_two_speaker_evidence(
    input_path: Path,
    run_dir: Path,
    diarization_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    separator_input = run_dir / "input_16k_mono.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(separator_input),
        ],
        check=True,
    )
    separation = _post_json(
        SEPARATOR_URL,
        {"audio_path": str(separator_input.resolve())},
    )
    rows = separation.get("speakers", [])
    if len(rows) != 2:
        raise RuntimeError(f"expected two separator outputs, received {len(rows)}")
    durations: dict[str, float] = {}
    for row in diarization_rows:
        label = str(row["speaker_id"])
        durations[label] = durations.get(label, 0.0) + max(
            0.0,
            float(row["end_seconds"]) - float(row["start_seconds"]),
        )
    ordered_labels = sorted(durations, key=lambda label: (-durations[label], label))
    outputs = []
    for index, row in enumerate(rows, start=1):
        source = Path(row["audio_path"])
        waveform, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        pyannote_label = ordered_labels[index - 1]
        spans = [
            [float(item["start_seconds"]), float(item["end_seconds"])]
            for item in diarization_rows
            if str(item["speaker_id"]) == pyannote_label
        ]
        mask = np.zeros(len(waveform), dtype=np.float32)
        compact_parts: list[np.ndarray] = []
        padding = 0.20
        fade = int(round(0.02 * sample_rate))
        for start_seconds, end_seconds in spans:
            left = max(0, int(round((start_seconds - padding) * sample_rate)))
            right = min(len(waveform), int(round((end_seconds + padding) * sample_rate)))
            if right <= left:
                continue
            local_mask = np.ones(right - left, dtype=np.float32)
            local_fade = min(fade, len(local_mask) // 2)
            if local_fade:
                ramp = np.linspace(0.0, 1.0, local_fade, dtype=np.float32)
                local_mask[:local_fade] *= ramp
                local_mask[-local_fade:] *= ramp[::-1]
            mask[left:right] = np.maximum(mask[left:right], local_mask)
            compact_parts.append(waveform[left:right] * local_mask)
        destination = run_dir / f"speaker_{index:02d}_evidence.wav"
        sf.write(destination, waveform * mask, sample_rate, subtype="PCM_16")
        compact_path = run_dir / f"speaker_{index:02d}_evidence_compact.wav"
        separator = np.zeros(int(round(0.10 * sample_rate)), dtype=np.float32)
        compact = (
            np.concatenate(
                [value for part in compact_parts for value in (part, separator)][:-1]
            )
            if compact_parts
            else np.zeros(int(round(0.25 * sample_rate)), dtype=np.float32)
        )
        sf.write(compact_path, compact, sample_rate, subtype="PCM_16")
        outputs.append(
            {
                "speaker_id": f"SPEAKER_{index:02d}",
                "pyannote_speaker_id": pyannote_label,
                "audio_path": str(destination.resolve()),
                "compact_audio_path": str(compact_path.resolve()),
                "evidence_spans_seconds": spans,
                "hypothesis": str(row.get("hypothesis") or ""),
                "speech_duration_seconds": float(row.get("speech_duration_seconds", 0.0)),
                "ordering": "dominant speech duration first",
            }
        )
    return {
        "applied": True,
        "method": "MossFormer2_SS_16K full-clip two-speaker separation",
        "speakers": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--separate-two-speaker", action="store_true")
    args = parser.parse_args()

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=True)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for input_path in args.input:
        input_path = input_path.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()[:16]
        run_dir = output_root / digest
        run_dir.mkdir(parents=True, exist_ok=True)
        output = pipeline(
            str(input_path),
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        regular = _tracks(output.speaker_diarization)
        exclusive = _tracks(output.exclusive_speaker_diarization)
        overlap = _overlap_regions(regular)
        labels = [str(value) for value in output.speaker_diarization.labels()]
        receipt: dict[str, Any] = {
            "format": "qces_pyannote_community1_evidence_v1",
            "complete": True,
            "input_path": str(input_path),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "model": MODEL_ID,
            "model_mode": "local inference",
            "annotations_used": False,
            "speaker_count": len(labels),
            "speaker_labels": labels,
            "regular_diarization": regular,
            "exclusive_diarization": exclusive,
            "overlap_regions": overlap,
            "speaker_embeddings": _embedding_audit(output),
            "evidence": {
                "applied": False,
                "reason": "not_requested",
                "speakers": [],
            },
        }
        if args.separate_two_speaker:
            if len(labels) == 2 and overlap:
                receipt["evidence"] = _render_two_speaker_evidence(
                    input_path,
                    run_dir,
                    regular,
                )
            else:
                receipt["evidence"] = {
                    "applied": False,
                    "reason": (
                        "requires exactly two diarized speakers and a detected overlap"
                    ),
                    "speakers": [],
                }
        receipt_path = run_dir / "receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary.append(
            {
                "input_path": str(input_path),
                "receipt_path": str(receipt_path.resolve()),
                "speaker_count": len(labels),
                "overlap_region_count": len(overlap),
                "evidence_applied": bool(receipt["evidence"]["applied"]),
            }
        )
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
