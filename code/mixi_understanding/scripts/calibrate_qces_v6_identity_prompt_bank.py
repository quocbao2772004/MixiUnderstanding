#!/usr/bin/env python3
"""Retext QCES-v6 stem caches with a calibrated identity prompt bank.

The expensive QCES-v6 cache already stores frozen AudioSep stems plus raw
AudioSep-CLAP audio embeddings for mixture windows and separated stems.  This
script does not re-run AudioSep.  It only changes the text vectors used by the
proposal head's identity channels:

* choose text prompt templates on train scenes by presence AUC;
* use the globally best train template as fallback for unseen/label-OOD classes;
* recompute ``clap_mixture_similarity`` and ``clap_stem_similarity`` in copied
  caches; and
* write a report so the experiment is auditable.

This tests whether the frozen separator-as-detector bottleneck is partly caused
by weak identity phrasing, before spending GPU on full multi-prompt separation
caches.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from mixi_understanding.scripts.cache_qces_v6_stem_features import (
    clap_text_embeddings,
    load_clap_encoder,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import describe


FORMAT_VERSION = "qces_v6_identity_prompt_bank_retext_v1"
DEFAULT_TEMPLATE_ID = "bare"
PROMPT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("bare", "{label}"),
    ("the_sound_of", "the sound of {label}"),
    ("sound_of", "sound of {label}"),
    ("label_sound", "{label} sound"),
    ("audio_of", "audio of {label}"),
    ("recording_of", "a recording of {label}"),
    ("audio_event", "the audio event {label}"),
    ("clear_sound", "a clear sound of {label}"),
    ("isolated_sound", "an isolated sound of {label}"),
    ("environmental_sound", "an environmental sound of {label}"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cache",
        action="append",
        nargs=3,
        metavar=("NAME", "INPUT_CACHE", "OUTPUT_CACHE"),
        required=True,
        help="Cache to retext. Repeat for train/val/test splits.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def scene_label_presence(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, set[str]]:
    by_scene: dict[str, set[str]] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        if scene_id in by_scene:
            continue
        labels: set[str] = set()
        for event in row["events"]:
            if event.get("event_kind") == "semantic":
                labels.add(str(event["label"]))
        by_scene[scene_id] = labels
    return by_scene


def prompt_text(template_id: str, label: str) -> str:
    template = dict(PROMPT_TEMPLATES)[template_id]
    text = describe(label).replace("_", " ").strip()
    return template.format(label=text)


def all_labels_from_caches(cache_paths: Sequence[Path]) -> tuple[str, ...]:
    labels: set[str] = set()
    for path in cache_paths:
        cache = torch.load(path, map_location="cpu", weights_only=False)
        for entry in cache["scenes"].values():
            labels.update(str(label) for label in entry["labels"])
    return tuple(sorted(labels))


def vector_for(
    *,
    label: str,
    template_id: str,
    prompt_vectors: Mapping[tuple[str, str], torch.Tensor],
) -> torch.Tensor:
    return prompt_vectors[(label, template_id)].float()


def score_rows_for_label(
    *,
    cache: Mapping[str, Any],
    presence: Mapping[str, set[str]],
    label: str,
    vector: torch.Tensor,
) -> tuple[list[int], list[float]]:
    targets: list[int] = []
    scores: list[float] = []
    text_vector = F.normalize(vector.float(), dim=-1)
    for scene_id, entry in cache["scenes"].items():
        labels = list(entry["labels"])
        if label not in labels:
            continue
        index = labels.index(label)
        target = int(label in presence.get(scene_id, set()))
        stem_vector = F.normalize(entry["clap_stem_vectors"][index].float(), dim=-1)
        window_vectors = F.normalize(entry["clap_window_vectors"].float(), dim=-1)
        stem_score = float(stem_vector @ text_vector)
        window_score = float((window_vectors @ text_vector).max())
        # The proposal head consumes both stem-level and window-level identity
        # channels, so template selection uses a simple joint score.
        scores.append(stem_score + window_score)
        targets.append(target)
    return targets, scores


def safe_auc(targets: Sequence[int], scores: Sequence[float]) -> float | None:
    if not targets or len(set(targets)) < 2:
        return None
    return float(roc_auc_score(targets, scores))


def select_templates(
    *,
    train_cache: Mapping[str, Any],
    presence: Mapping[str, set[str]],
    labels: Sequence[str],
    prompt_vectors: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    template_scores: dict[str, list[float]] = defaultdict(list)
    label_reports: dict[str, Any] = {}
    template_ids = [template_id for template_id, _ in PROMPT_TEMPLATES]

    for label in labels:
        per_template: dict[str, Any] = {}
        for template_id in template_ids:
            targets, scores = score_rows_for_label(
                cache=train_cache,
                presence=presence,
                label=label,
                vector=vector_for(
                    label=label,
                    template_id=template_id,
                    prompt_vectors=prompt_vectors,
                ),
            )
            auc = safe_auc(targets, scores)
            per_template[template_id] = {
                "auc": auc,
                "examples": len(targets),
                "positives": int(sum(targets)),
                "prompt": prompt_text(template_id, label),
            }
            if auc is not None:
                template_scores[template_id].append(auc)

        valid = [
            (template_id, info["auc"])
            for template_id, info in per_template.items()
            if info["auc"] is not None
        ]
        if valid:
            selected_template, selected_auc = max(
                valid, key=lambda item: (item[1], item[0] == DEFAULT_TEMPLATE_ID)
            )
            default_auc = per_template[DEFAULT_TEMPLATE_ID]["auc"]
            reason = "per_label_train_auc"
        else:
            selected_template = DEFAULT_TEMPLATE_ID
            selected_auc = None
            default_auc = None
            reason = "no_train_auc"
        label_reports[label] = {
            "selected_template": selected_template,
            "selected_prompt": prompt_text(selected_template, label),
            "selected_auc": selected_auc,
            "default_auc": default_auc,
            "reason": reason,
            "templates": per_template,
        }

    global_template_summary = {
        template_id: {
            "macro_auc": (
                float(sum(values) / len(values)) if values else None
            ),
            "num_labels": len(values),
        }
        for template_id, values in template_scores.items()
    }
    valid_global = [
        (template_id, info["macro_auc"])
        for template_id, info in global_template_summary.items()
        if info["macro_auc"] is not None
    ]
    global_template = (
        max(valid_global, key=lambda item: (item[1], item[0] == DEFAULT_TEMPLATE_ID))[0]
        if valid_global
        else DEFAULT_TEMPLATE_ID
    )

    # Replace labels without a valid per-label AUC, which mostly covers
    # label-OOD/unseen cache labels, with the train-selected global template.
    for label, report in label_reports.items():
        if report["reason"] == "no_train_auc":
            report["selected_template"] = global_template
            report["selected_prompt"] = prompt_text(global_template, label)
            report["reason"] = "global_train_auc_fallback"

    default_aucs = [
        report["default_auc"]
        for report in label_reports.values()
        if report["default_auc"] is not None
    ]
    selected_aucs = [
        report["selected_auc"]
        for report in label_reports.values()
        if report["selected_auc"] is not None
    ]
    return {
        "global_template": global_template,
        "global_template_summary": global_template_summary,
        "default_macro_auc": float(sum(default_aucs) / len(default_aucs))
        if default_aucs
        else None,
        "selected_macro_auc": float(sum(selected_aucs) / len(selected_aucs))
        if selected_aucs
        else None,
        "labels": label_reports,
    }


def retext_cache(
    *,
    input_path: Path,
    output_path: Path,
    selection: Mapping[str, Any],
    prompt_vectors: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, Any]:
    cache = torch.load(input_path, map_location="cpu", weights_only=False)
    num_frames = int(cache["num_frames"])
    new_cache = copy.copy(cache)
    new_scenes: dict[str, Any] = {}
    for scene_id, entry in cache["scenes"].items():
        labels = [str(label) for label in entry["labels"]]
        vectors = []
        prompts = []
        for label in labels:
            label_selection = selection["labels"].get(label)
            template_id = (
                label_selection["selected_template"]
                if label_selection is not None
                else selection["global_template"]
            )
            vectors.append(
                vector_for(
                    label=label,
                    template_id=template_id,
                    prompt_vectors=prompt_vectors,
                )
            )
            prompts.append(prompt_text(template_id, label))
        label_vectors = F.normalize(torch.stack(vectors).float(), dim=-1)
        window_vectors = F.normalize(entry["clap_window_vectors"].float(), dim=-1)
        stem_vectors = F.normalize(entry["clap_stem_vectors"].float(), dim=-1)
        window_similarity = (label_vectors @ window_vectors.T)[None]
        mixture_similarity = F.interpolate(
            window_similarity,
            size=num_frames,
            mode="linear",
            align_corners=True,
        )[0]
        stem_similarity = stem_vectors @ label_vectors.T

        new_entry = copy.copy(entry)
        new_entry["clap_mixture_similarity"] = mixture_similarity.half().cpu()
        new_entry["clap_stem_similarity"] = stem_similarity.float().cpu()
        new_entry["identity_prompts"] = prompts
        new_scenes[scene_id] = new_entry

    new_cache["format"] = FORMAT_VERSION
    new_cache["source_cache"] = str(input_path.resolve())
    new_cache["source_cache_sha256"] = sha256_file(input_path.resolve())
    new_cache["identity_prompt_bank"] = {
        "format": FORMAT_VERSION,
        "global_template": selection["global_template"],
        "prompt_templates": dict(PROMPT_TEMPLATES),
    }
    new_cache["scenes"] = new_scenes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_cache, output_path)
    return {
        "input_cache": str(input_path.resolve()),
        "output_cache": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path.resolve()),
        "scenes": len(new_scenes),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# QCES-v6 identity prompt-bank calibration",
        "",
        "This experiment does not re-run AudioSep. It reuses cached separator "
        "stems and recalibrates only the AudioSep-CLAP text vectors used by the "
        "proposal head identity channels.",
        "",
        f"Global fallback template: `{report['selection']['global_template']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| default bare-label macro AUC ↑ | {report['selection']['default_macro_auc']:.4f} |",
        f"| selected per-label macro AUC ↑ | {report['selection']['selected_macro_auc']:.4f} |",
        "",
        "## Global template macro AUC",
        "",
        "| Template | Macro AUC ↑ | Labels | Example |",
        "|---|---:|---:|---|",
    ]
    for template_id, info in sorted(
        report["selection"]["global_template_summary"].items(),
        key=lambda item: (item[1]["macro_auc"] is None, -(item[1]["macro_auc"] or -1)),
    ):
        example = prompt_text(template_id, "Dog")
        auc = info["macro_auc"]
        lines.append(
            f"| `{template_id}` | {auc:.4f} | {info['num_labels']} | `{example}` |"
            if auc is not None
            else f"| `{template_id}` | -- | {info['num_labels']} | `{example}` |"
        )
    changed = [
        (label, item)
        for label, item in report["selection"]["labels"].items()
        if item["selected_template"] != DEFAULT_TEMPLATE_ID
    ]
    lines.extend(
        [
            "",
            f"Changed labels: {len(changed)} / {len(report['selection']['labels'])}",
            "",
            "| Label | Selected template | Default AUC ↑ | Selected AUC ↑ | Prompt |",
            "|---|---|---:|---:|---|",
        ]
    )
    for label, item in sorted(changed)[:80]:
        default_auc = item["default_auc"]
        selected_auc = item["selected_auc"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"`{item['selected_template']}`",
                    f"{default_auc:.4f}" if default_auc is not None else "--",
                    f"{selected_auc:.4f}" if selected_auc is not None else "--",
                    f"`{item['selected_prompt']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Written caches",
            "",
            "| Split | Scenes | Output |",
            "|---|---:|---|",
        ]
    )
    for name, item in report["written_caches"].items():
        lines.append(f"| {name} | {item['scenes']} | `{item['output_cache']}` |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_specs = [
        (name, Path(input_cache).resolve(), Path(output_cache).resolve())
        for name, input_cache, output_cache in args.cache
    ]
    labels = all_labels_from_caches([path for _, path, _ in cache_specs])
    all_prompts = {
        prompt_text(template_id, label)
        for label in labels
        for template_id, _ in PROMPT_TEMPLATES
    }

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    encoder = load_clap_encoder(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), device
    )
    encoded = {
        prompt: F.normalize(vector.float(), dim=-1).cpu()
        for prompt, vector in clap_text_embeddings(
            encoder, sorted(all_prompts), args.text_batch_size
        ).items()
    }
    prompt_vectors = {
        (label, template_id): encoded[prompt_text(template_id, label)]
        for label in labels
        for template_id, _ in PROMPT_TEMPLATES
    }

    train_cache = torch.load(
        args.train_cache.resolve(), map_location="cpu", weights_only=False
    )
    train_rows = read_manifest(args.train_manifest.resolve())
    selection = select_templates(
        train_cache=train_cache,
        presence=scene_label_presence(train_rows),
        labels=labels,
        prompt_vectors=prompt_vectors,
    )

    written: dict[str, Any] = {}
    for name, input_path, output_path in cache_specs:
        written[name] = retext_cache(
            input_path=input_path,
            output_path=output_path,
            selection=selection,
            prompt_vectors=prompt_vectors,
        )

    report = {
        "format": FORMAT_VERSION,
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "train_cache_sha256": sha256_file(args.train_cache.resolve()),
        "audiosep_root": str(args.audiosep_root.resolve()),
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "device": str(device),
        "prompt_templates": dict(PROMPT_TEMPLATES),
        "selection": selection,
        "written_caches": written,
    }
    (output_dir / "identity_prompt_bank_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "identity_prompt_bank_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )
    print(output_dir / "identity_prompt_bank_report.md")


if __name__ == "__main__":
    main()

