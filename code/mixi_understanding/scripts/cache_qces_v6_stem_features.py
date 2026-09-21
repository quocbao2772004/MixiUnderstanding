#!/usr/bin/env python3
"""Cache frozen-separator stem descriptors for every QCES scene.

For each scene this queries the frozen text-queried separator once per
candidate label that any of the scene's questions can name, and stores a compact
frame-level descriptor of the returned stem.  Nothing about the label identity
is stored, so a proposal reader trained on the cache cannot memorise a closed
label set.

Candidate labels come from the question surface string and the answer options
only, through the QCES-v6 parser.  The scene's event annotation is never read
here.
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
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch

from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.stem_features import (
    NUM_FRAMES,
    STEM_FEATURE_DIM,
    waveform_frame_features,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    describe,
)

FORMAT_VERSION = "qces_v6_stem_feature_cache_v2"
CLAP_WINDOW_SECONDS = 1.0
CLAP_HOP_SECONDS = 0.25


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--detection-prompt",
        default="the sound of {label}",
        help="Template used for the per-candidate detection query.",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_clap_encoder(repository_root: Path, checkpoint_path: Path, device):
    """Load AudioSep's own frozen query encoder and keep it resident.

    ``encode_prompts`` builds and discards this encoder per call.  The proposal
    cache needs both the text and the audio side of the same embedding space, so
    the encoder is loaded once here with the identical weight-selection rule.
    """

    root_string = str(repository_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from models.clap_encoder import CLAP_Encoder

    encoder = CLAP_Encoder(pretrained_path="").eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("query_encoder."): value
        for key, value in payload.items()
        if key.startswith("query_encoder.")
    }
    incompatible = encoder.load_state_dict(state, strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith("embeddings.position_ids")
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "incompatible AudioSep CLAP state: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    return encoder.to(device).eval()


def clap_text_embeddings(
    encoder, prompts: Sequence[str], batch_size: int
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            embeddings = encoder.get_query_embed(modality="text", text=batch)
            result.update(dict(zip(batch, embeddings.detach().cpu())))
    return result


def clap_audio_embeddings(
    encoder, waveforms: torch.Tensor, batch_size: int = 32
) -> torch.Tensor:
    """Embed ``[B, samples]`` audio in the shared space, batched."""

    chunks = []
    with torch.inference_mode():
        for start in range(0, waveforms.shape[0], batch_size):
            chunk = waveforms[start : start + batch_size]
            chunks.append(
                encoder.get_query_embed(modality="audio", audio=chunk).detach()
            )
    return torch.cat(chunks, dim=0)


def sliding_windows(
    waveform: torch.Tensor, window: int, hop: int
) -> torch.Tensor:
    if waveform.numel() < window:
        waveform = torch.nn.functional.pad(
            waveform, (0, window - waveform.numel())
        )
    return waveform.unfold(0, window, hop).contiguous()


def load_taxonomy(dataset_config: Path) -> tuple[str, ...]:
    payload = json.loads(dataset_config.read_text(encoding="utf-8"))
    composition = payload["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


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
    taxonomy = load_taxonomy(args.dataset_config.resolve())

    scenes: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
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
            parsed = parse_question(
                row["question"], row["answer_options"], taxonomy
            )
            for label in parsed.query_labels:
                if label not in entry["labels"]:
                    entry["labels"].append(label)
    if args.max_scenes:
        scenes = OrderedDict(list(scenes.items())[: args.max_scenes])
    if not scenes:
        raise SystemExit("manifest produced no scenes")

    required_prompts = sorted(
        {
            args.detection_prompt.format(label=describe(label).replace("_", " "))
            for entry in scenes.values()
            for label in entry["labels"]
        }
    )
    print(
        f"scenes={len(scenes)} unique_detection_prompts={len(required_prompts)}",
        flush=True,
    )
    encoder = load_clap_encoder(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), device
    )
    embeddings = clap_text_embeddings(
        encoder, required_prompts, args.text_batch_size
    )
    # The identity channel compares audio against a bare class phrase rather
    # than the separation query, because the separation query is a request and
    # the identity question is "is this class here at all".
    label_prompts = sorted(
        {
            describe(label).replace("_", " ")
            for entry in scenes.values()
            for label in entry["labels"]
        }
    )
    label_embeddings = {
        prompt: torch.nn.functional.normalize(vector, dim=-1)
        for prompt, vector in clap_text_embeddings(
            encoder, label_prompts, args.text_batch_size
        ).items()
    }
    separator = _load_separator(args, device)
    window = int(round(CLAP_WINDOW_SECONDS * 32_000))
    hop = int(round(CLAP_HOP_SECONDS * 32_000))

    cache: dict[str, dict[str, torch.Tensor | list[str]]] = {}
    separator_calls = 0
    started = time.time()
    with torch.inference_mode():
        for index, (scene_id, entry) in enumerate(scenes.items(), start=1):
            waveform, sample_rate = sf.read(
                dataset_root / entry["mixture_path"], dtype="float32", always_2d=False
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
            for row, label in enumerate(labels):
                prompt = args.detection_prompt.format(
                    label=describe(label).replace("_", " ")
                )
                condition = embeddings[prompt][None].to(device)
                stem = separator(
                    {"mixture": mixture[None, None], "condition": condition}
                )["waveform"][0, 0]
                separator_calls += 1
                stems[row] = waveform_frame_features(stem).half().cpu()
                stem_waveforms[row] = stem

            label_vectors = torch.stack(
                [
                    label_embeddings[describe(label).replace("_", " ")]
                    for label in labels
                ]
            ).to(device)
            windows = sliding_windows(mixture, window, hop)
            window_vectors = torch.nn.functional.normalize(
                clap_audio_embeddings(encoder, windows), dim=-1
            )
            # [labels, windows] resampled onto the shared frame grid.
            window_similarity = (label_vectors @ window_vectors.T)[None]
            mixture_similarity = torch.nn.functional.interpolate(
                window_similarity, size=NUM_FRAMES, mode="linear", align_corners=True
            )[0]
            stem_vectors = torch.nn.functional.normalize(
                clap_audio_embeddings(encoder, stem_waveforms), dim=-1
            )

            cache[scene_id] = {
                "labels": labels,
                "stems": stems,
                "mixture": waveform_frame_features(mixture).half().cpu(),
                "clap_mixture_similarity": mixture_similarity.half().cpu(),
                "clap_stem_similarity": (stem_vectors @ label_vectors.T).cpu(),
                # The raw unit-norm embeddings are ~60 kB per scene and let any
                # later identity experiment -- a different label phrasing, a
                # prompt ensemble, a per-class calibration -- be run from the
                # cache instead of from the GPU separator again.
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
            "detection_prompt": args.detection_prompt,
            "num_frames": NUM_FRAMES,
            "stem_feature_dim": STEM_FEATURE_DIM,
            "separator_calls": separator_calls,
            "clap_window_seconds": CLAP_WINDOW_SECONDS,
            "clap_hop_seconds": CLAP_HOP_SECONDS,
            "label_text_vectors": {
                label: vector.half()
                for label, vector in label_embeddings.items()
            },
            "scenes": cache,
        },
        output,
    )
    print(f"wrote {output} scenes={len(cache)} calls={separator_calls}", flush=True)


if __name__ == "__main__":
    main()
