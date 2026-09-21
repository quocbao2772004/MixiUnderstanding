#!/usr/bin/env python3
"""Cache QCES-v6 stem descriptors with per-label prompt-bank AudioSep queries.

This is a follow-up to ``calibrate_qces_v6_identity_prompt_bank.py``.  The
identity retext experiment changes only CLAP text vectors in an existing cache;
this script spends GPU again and asks the frozen separator with the selected
per-label prompt itself.  It intentionally lives in a new file so the original
cache builder stays unchanged.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F

from mixi_understanding.qces.question_parsing import parse_question
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
    load_taxonomy,
    sliding_windows,
)
from mixi_understanding.scripts.calibrate_qces_v6_identity_prompt_bank import (
    prompt_text,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import _load_separator


FORMAT_VERSION = "qces_v6_stem_feature_cache_promptbank_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-bank-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_prompts(report_path: Path) -> tuple[dict[str, str], str]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    selection = payload["selection"]
    global_template = str(selection.get("global_template", "the_sound_of"))
    labels = selection.get("labels", {})
    result: dict[str, str] = {}
    for label, info in labels.items():
        prompt = info.get("selected_prompt")
        if isinstance(prompt, str) and prompt.strip():
            result[str(label)] = prompt.strip()
    return result, global_template


def prompt_for_label(
    label: str, prompt_map: Mapping[str, str], global_template: str
) -> str:
    prompt = prompt_map.get(label)
    if prompt:
        return prompt
    return prompt_text(global_template, label)


def read_scene_requests(
    manifest: Path, dataset_config: Path, max_scenes: int
) -> OrderedDict[str, dict[str, Any]]:
    taxonomy = load_taxonomy(dataset_config)
    scenes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scene_id = row["scene_id"]
            entry = scenes.get(scene_id)
            if entry is None:
                entry = {
                    "mixture_path": row["mixture_path"],
                    "sample_rate": int(row["sample_rate"]),
                    "labels": [],
                    "record_ids": [],
                }
                scenes[scene_id] = entry
            entry["record_ids"].append(row["id"])
            parsed = parse_question(row["question"], row["answer_options"], taxonomy)
            for label in parsed.query_labels:
                if label not in entry["labels"]:
                    entry["labels"].append(label)
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
    prompt_map, global_template = selected_prompts(args.prompt_bank_report.resolve())
    scenes = read_scene_requests(
        manifest, args.dataset_config.resolve(), max_scenes=args.max_scenes
    )

    label_prompt_map = {
        label: prompt_for_label(label, prompt_map, global_template)
        for entry in scenes.values()
        for label in entry["labels"]
    }
    required_prompts = sorted(set(label_prompt_map.values()))
    print(
        f"scenes={len(scenes)} unique_promptbank_prompts={len(required_prompts)}",
        flush=True,
    )

    encoder = load_clap_encoder(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), device
    )
    text_embeddings = {
        prompt: F.normalize(vector, dim=-1)
        for prompt, vector in clap_text_embeddings(
            encoder, required_prompts, args.text_batch_size
        ).items()
    }
    separator = _load_separator(args, device)
    window = int(round(CLAP_WINDOW_SECONDS * 32_000))
    hop = int(round(CLAP_HOP_SECONDS * 32_000))

    cache: dict[str, dict[str, torch.Tensor | list[str] | dict[str, str]]] = {}
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
                    f"{scene_id}: manifest sample rate {entry['sample_rate']} "
                    f"differs from file {sample_rate}"
                )
            mixture = torch.from_numpy(waveform).to(device)
            if mixture.ndim != 1:
                mixture = mixture.mean(dim=-1)

            labels = list(entry["labels"])
            stems = torch.zeros(
                (len(labels), STEM_FEATURE_DIM, NUM_FRAMES), dtype=torch.float16
            )
            stem_waveforms = torch.zeros(
                (len(labels), mixture.numel()), device=device
            )
            prompts: dict[str, str] = {}
            for row, label in enumerate(labels):
                prompt = label_prompt_map[label]
                prompts[label] = prompt
                condition = text_embeddings[prompt][None].to(device)
                stem = separator(
                    {"mixture": mixture[None, None], "condition": condition}
                )["waveform"][0, 0]
                separator_calls += 1
                stems[row] = waveform_frame_features(stem).half().cpu()
                stem_waveforms[row] = stem

            label_vectors = torch.stack(
                [text_embeddings[label_prompt_map[label]] for label in labels]
            ).to(device)
            windows = sliding_windows(mixture, window, hop)
            window_vectors = F.normalize(clap_audio_embeddings(encoder, windows), dim=-1)
            window_similarity = (label_vectors @ window_vectors.T)[None]
            mixture_similarity = F.interpolate(
                window_similarity, size=NUM_FRAMES, mode="linear", align_corners=True
            )[0]
            stem_vectors = F.normalize(
                clap_audio_embeddings(encoder, stem_waveforms), dim=-1
            )

            cache[scene_id] = {
                "labels": labels,
                "label_prompts": prompts,
                "stems": stems,
                "mixture": waveform_frame_features(mixture).half().cpu(),
                "clap_mixture_similarity": mixture_similarity.half().cpu(),
                "clap_stem_similarity": (stem_vectors @ label_vectors.T).cpu(),
                "clap_window_vectors": window_vectors.half().cpu(),
                "clap_stem_vectors": stem_vectors.half().cpu(),
            }
            del stem_waveforms
            if index % 20 == 0 or index == len(scenes):
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-6)
                print(
                    f"scene {index}/{len(scenes)} calls={separator_calls} "
                    f"{rate:.2f} scene/s eta={(len(scenes) - index) / rate / 60:.1f} min",
                    flush=True,
                )

    torch.save(
        {
            "format": FORMAT_VERSION,
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
            "audiosep_checkpoint_sha256": sha256_file(
                args.audiosep_checkpoint.resolve()
            ),
            "prompt_bank_report": str(args.prompt_bank_report.resolve()),
            "prompt_bank_report_sha256": sha256_file(args.prompt_bank_report.resolve()),
            "global_template": global_template,
            "label_detection_prompts": label_prompt_map,
            "num_frames": NUM_FRAMES,
            "stem_feature_dim": STEM_FEATURE_DIM,
            "separator_calls": separator_calls,
            "clap_window_seconds": CLAP_WINDOW_SECONDS,
            "clap_hop_seconds": CLAP_HOP_SECONDS,
            "label_text_vectors": {
                label: text_embeddings[prompt].half().cpu()
                for label, prompt in label_prompt_map.items()
            },
            "scenes": cache,
        },
        output,
    )
    print(f"wrote {output} scenes={len(cache)} calls={separator_calls}", flush=True)


if __name__ == "__main__":
    main()
