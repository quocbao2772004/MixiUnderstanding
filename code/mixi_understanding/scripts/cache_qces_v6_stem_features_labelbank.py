#!/usr/bin/env python3
"""Cache QCES-v6 stem descriptors for a fixed open-event label bank.

The original v6 cache only queries labels mentioned by each scene's benchmark
questions/options.  This sidecar probes scalability by querying the same fixed
label bank for every scene, e.g. 100--200 AudioSet/FSD50K class prompts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F

from mixi_understanding.qces.stem_features import (
    NUM_FRAMES,
    STEM_FEATURE_DIM,
    waveform_frame_features,
)
from mixi_understanding.scripts.cache_qces_v6_stem_features import (
    CLAP_HOP_SECONDS,
    CLAP_WINDOW_SECONDS,
    clap_audio_embeddings,
    clap_text_embeddings,
    load_clap_encoder,
    sliding_windows,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    describe,
)


FORMAT_VERSION = "qces_v6_stem_feature_cache_fixed_labelbank_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label-bank", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument(
        "--detection-prompt",
        default="the sound of {label}",
        help="Template used for each open-event candidate label.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_bank(path: Path, max_labels: int) -> tuple[str, ...]:
    labels: list[str] = []
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("labels", payload) if isinstance(payload, dict) else payload
        for item in raw:
            label = str(item.get("label", item) if isinstance(item, dict) else item).strip()
            if label and label not in labels:
                labels.append(label)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                label = (row.get("label") or row.get("display_name") or row.get("name") or "").strip()
                if label and label not in labels:
                    labels.append(label)
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                label = line.strip()
                if label and not label.startswith("#") and label not in labels:
                    labels.append(label)
    if max_labels:
        labels = labels[:max_labels]
    if not labels:
        raise SystemExit(f"label bank is empty: {path}")
    return tuple(labels)


def read_scenes(manifest: Path, max_scenes: int) -> OrderedDict[str, dict[str, Any]]:
    scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            scene_id = row["scene_id"]
            if scene_id not in scenes:
                scenes[scene_id] = {
                    "mixture_path": row["mixture_path"],
                    "sample_rate": int(row["sample_rate"]),
                    "record_ids": [],
                }
            scenes[scene_id]["record_ids"].append(row["id"])
    if max_scenes:
        scenes = OrderedDict(list(scenes.items())[:max_scenes])
    if not scenes:
        raise SystemExit("manifest produced no scenes")
    return scenes


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    manifest = args.manifest.resolve()
    dataset_root = manifest.parent
    labels = read_label_bank(args.label_bank.resolve(), args.max_labels)
    scenes = read_scenes(manifest, args.max_scenes)
    prompts = {
        label: args.detection_prompt.format(label=describe(label).replace("_", " "))
        for label in labels
    }
    print(
        f"scenes={len(scenes)} labels={len(labels)} separator_calls={len(scenes) * len(labels)}",
        flush=True,
    )

    encoder = load_clap_encoder(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), device
    )
    prompt_vectors = {
        prompt: F.normalize(vector, dim=-1)
        for prompt, vector in clap_text_embeddings(
            encoder, sorted(set(prompts.values())), args.text_batch_size
        ).items()
    }
    separator = _load_separator(args, device)
    window = int(round(CLAP_WINDOW_SECONDS * 32_000))
    hop = int(round(CLAP_HOP_SECONDS * 32_000))

    cache: dict[str, dict[str, Any]] = {}
    separator_calls = 0
    started = time.time()
    with torch.inference_mode():
        for index, (scene_id, entry) in enumerate(scenes.items(), start=1):
            waveform, sample_rate = sf.read(
                dataset_root / entry["mixture_path"],
                dtype="float32",
                always_2d=False,
            )
            if sample_rate != entry["sample_rate"]:
                raise SystemExit(
                    f"{scene_id}: sample-rate mismatch manifest={entry['sample_rate']} wav={sample_rate}"
                )
            mixture = torch.from_numpy(waveform).to(device)
            if mixture.ndim != 1:
                mixture = mixture.mean(dim=-1)

            stems = torch.zeros(
                (len(labels), STEM_FEATURE_DIM, NUM_FRAMES), dtype=torch.float16
            )
            stem_waveforms = torch.zeros(
                (len(labels), mixture.numel()), device=device
            )
            for row, label in enumerate(labels):
                condition = prompt_vectors[prompts[label]][None].to(device)
                stem = separator({"mixture": mixture[None, None], "condition": condition})[
                    "waveform"
                ][0, 0]
                separator_calls += 1
                stems[row] = waveform_frame_features(stem).half().cpu()
                stem_waveforms[row] = stem

            label_vectors = torch.stack([prompt_vectors[prompts[label]] for label in labels]).to(device)
            windows = sliding_windows(mixture, window, hop)
            window_vectors = F.normalize(clap_audio_embeddings(encoder, windows), dim=-1)
            mixture_similarity = F.interpolate(
                (label_vectors @ window_vectors.T)[None],
                size=NUM_FRAMES,
                mode="linear",
                align_corners=True,
            )[0]
            stem_vectors = F.normalize(clap_audio_embeddings(encoder, stem_waveforms), dim=-1)

            cache[scene_id] = {
                "labels": list(labels),
                "label_prompts": prompts,
                "stems": stems,
                "mixture": waveform_frame_features(mixture).half().cpu(),
                "clap_mixture_similarity": mixture_similarity.half().cpu(),
                "clap_stem_similarity": (stem_vectors @ label_vectors.T).cpu(),
                "clap_window_vectors": window_vectors.half().cpu(),
                "clap_stem_vectors": stem_vectors.half().cpu(),
            }
            del stem_waveforms
            elapsed = time.time() - started
            rate = index / max(elapsed, 1e-6)
            print(
                f"scene {index}/{len(scenes)} calls={separator_calls} "
                f"{rate:.3f} scene/s eta={(len(scenes) - index) / max(rate, 1e-6) / 60:.1f} min",
                flush=True,
            )

    torch.save(
        {
            "format": FORMAT_VERSION,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "label_bank": str(args.label_bank.resolve()),
            "label_bank_sha256": sha256_file(args.label_bank.resolve()),
            "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
            "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
            "detection_prompt": args.detection_prompt,
            "num_frames": NUM_FRAMES,
            "stem_feature_dim": STEM_FEATURE_DIM,
            "separator_calls": separator_calls,
            "labels": list(labels),
            "scenes": cache,
        },
        output,
    )
    print(f"wrote {output} scenes={len(cache)} labels={len(labels)} calls={separator_calls}", flush=True)


if __name__ == "__main__":
    main()
