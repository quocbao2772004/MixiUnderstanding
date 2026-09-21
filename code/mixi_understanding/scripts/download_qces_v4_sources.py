#!/usr/bin/env python3
"""Download the deterministic AudioTime source pool used by QCES v4."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.build_qa_removal_dataset import (  # noqa: E402
    SourceClip,
    load_sources,
    stable_seed,
)
from mixi_understanding.scripts.download_audiotime_subset import (  # noqa: E402
    DATASET_NAME,
    download_file,
    request_json,
    request_url_json,
    sha256_file,
    validate_audio,
)


PINNED_DATASET_REVISION = "50e1a871c03eee734af88ee2b61e13a372a38a8d"
DEFAULT_SEED = 271_828
SOURCES_PER_LABEL = 4
SEMANTIC_MINIMUM_INTERVAL_SECONDS = 1.5
NUISANCE_MINIMUM_INTERVAL_SECONDS = 5.8

SEMANTIC_LABELS: Tuple[str, ...] = (
    "Croak",
    "Engine knocking",
    "Jackhammer",
    "Chainsaw",
    "Vacuum cleaner",
    "Fire alarm",
    "Mechanical bell",
    "Steam whistle",
    "Helicopter",
    "Printer",
    "Wind chime",
    "Rain",
)
NUISANCE_LABELS: Tuple[str, ...] = (
    "Ambulance (siren)",
    "Sawing",
    "Mechanical fan",
)


@dataclass(frozen=True)
class SelectedSource:
    """One source selected for a fixed QCES v4 role and label."""

    role: str
    source: SourceClip


def bounded_page_size(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be in [1, 100]")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must not be empty")
    return value.strip()


def git_commit(value: str) -> str:
    parsed = value.strip().lower()
    if len(parsed) != 40 or any(
        character not in "0123456789abcdef" for character in parsed
    ):
        raise argparse.ArgumentTypeError("must be a full 40-character Git commit")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument("--dataset", type=nonempty, default=DATASET_NAME)
    parser.add_argument("--revision", type=git_commit, default=PINNED_DATASET_REVISION)
    parser.add_argument("--config", type=nonempty, default="default")
    parser.add_argument("--split", type=nonempty, default="train")
    parser.add_argument("--seed", type=nonnegative_int, default=DEFAULT_SEED)
    parser.add_argument("--page-size", type=bounded_page_size, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _source_sort_key(source: SourceClip) -> Tuple[str, int, str]:
    prefix, separator, suffix = source.source_id.rpartition("_")
    numeric_suffix = int(suffix) if separator and suffix.isdigit() else -1
    return prefix if separator else source.source_id, numeric_suffix, source.source_id


def _select_label_sources(
    candidates: Sequence[SourceClip],
    *,
    label: str,
    role: str,
    minimum_interval_seconds: float,
    seed: int,
) -> List[SelectedSource]:
    eligible = sorted(
        (
            source
            for source in candidates
            if source.label == label
            and source.duration_seconds + 1e-9 >= minimum_interval_seconds
        ),
        key=_source_sort_key,
    )
    if len(eligible) < SOURCES_PER_LABEL:
        raise RuntimeError(
            f"{label!r} has {len(eligible)} clean AudioTime recordings with an "
            f"interval >= {minimum_interval_seconds:.1f}s; need {SOURCES_PER_LABEL}"
        )
    random.Random(stable_seed(seed, "qces-v4-source-order", role, label)).shuffle(
        eligible
    )
    return [
        SelectedSource(role=role, source=source)
        for source in eligible[:SOURCES_PER_LABEL]
    ]


def selected_sources(audiotime_root: Path, seed: int) -> List[SelectedSource]:
    """Select four clean recordings per fixed label, independently by label."""

    # load_sources intentionally keeps only metadata rows containing one label and
    # one interval. This avoids blessing a second foreground event as part of a
    # supposedly isolated source stem.
    candidates = load_sources(audiotime_root, require_audio=False)
    selected: List[SelectedSource] = []
    for label in SEMANTIC_LABELS:
        selected.extend(
            _select_label_sources(
                candidates,
                label=label,
                role="semantic",
                minimum_interval_seconds=SEMANTIC_MINIMUM_INTERVAL_SECONDS,
                seed=seed,
            )
        )
    for label in NUISANCE_LABELS:
        selected.extend(
            _select_label_sources(
                candidates,
                label=label,
                role="nuisance",
                minimum_interval_seconds=NUISANCE_MINIMUM_INTERVAL_SECONDS,
                seed=seed,
            )
        )
    validate_selection(selected)
    return selected


def validate_selection(selected: Sequence[SelectedSource]) -> None:
    expected_count = SOURCES_PER_LABEL * (
        len(SEMANTIC_LABELS) + len(NUISANCE_LABELS)
    )
    if len(selected) != expected_count:
        raise ValueError(
            f"selection has {len(selected)} sources; need {expected_count}"
        )

    source_ids = [item.source.source_id for item in selected]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("selection reuses an AudioTime recording")

    expected: Mapping[str, Tuple[Tuple[str, ...], float]] = {
        "semantic": (SEMANTIC_LABELS, SEMANTIC_MINIMUM_INTERVAL_SECONDS),
        "nuisance": (NUISANCE_LABELS, NUISANCE_MINIMUM_INTERVAL_SECONDS),
    }
    for role, (labels, minimum_duration) in expected.items():
        for label in labels:
            matching = [
                item
                for item in selected
                if item.role == role and item.source.label == label
            ]
            if len(matching) != SOURCES_PER_LABEL:
                raise ValueError(
                    f"selection has {len(matching)} {role} sources for {label!r}; "
                    f"need {SOURCES_PER_LABEL}"
                )
            if any(
                item.source.duration_seconds + 1e-9 < minimum_duration
                for item in matching
            ):
                raise ValueError(f"selection contains an undersized {role} interval")

    allowed_pairs = {
        (role, label)
        for role, (labels, _) in expected.items()
        for label in labels
    }
    unexpected = sorted(
        {
            (item.role, item.source.label)
            for item in selected
            if (item.role, item.source.label) not in allowed_pairs
        }
    )
    if unexpected:
        raise ValueError(
            f"selection contains unexpected role/label pairs: {unexpected}"
        )


def verify_pinned_revision(dataset: str, revision: str) -> None:
    revision_url = (
        "https://huggingface.co/api/datasets/"
        + urllib.parse.quote(dataset, safe="/")
        + "/revision/"
        + urllib.parse.quote(revision, safe="")
    )
    dataset_info = request_url_json(revision_url)
    resolved_revision = dataset_info.get("sha")
    if resolved_revision != revision:
        raise RuntimeError(
            f"Hugging Face resolved revision {resolved_revision!r}, "
            f"expected {revision!r}"
        )


def discover_pinned_assets(
    source_ids: Sequence[str],
    *,
    dataset: str,
    revision: str,
    config: str,
    split: str,
    page_size: int,
    cache_path: Path,
) -> Dict[str, str]:
    wanted = sorted(set(source_ids))
    identity = {
        "format": "qces_v4_asset_discovery_v1",
        "dataset": dataset,
        "revision": revision,
        "config": config,
        "split": split,
        "page_size": page_size,
        "source_ids": wanted,
    }
    offset = 0
    assets: Dict[str, str] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in identity.items()):
            offset = int(cached.get("next_offset", 0))
            cached_assets = cached.get("assets", {})
            if isinstance(cached_assets, dict) and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in cached_assets.items()
            ):
                assets = dict(cached_assets)
                print(
                    f"resuming asset discovery at row {offset}; "
                    f"found {len(assets)}/{len(wanted)} sources"
                )
            else:
                offset = 0
                assets = {}

    while set(wanted) - set(assets):
        payload = request_json(
            "/rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            },
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for wrapped in rows:
            row = wrapped.get("row", {})
            source_id = row.get("audio_id")
            if source_id not in wanted:
                continue
            audio = row.get("audio")
            if not isinstance(audio, list) or not audio:
                raise ValueError(f"missing audio asset for {source_id}")
            source_url = audio[0].get("src")
            if not isinstance(source_url, str) or not source_url:
                raise ValueError(f"missing audio URL for {source_id}")
            assets[source_id] = source_url
        offset += len(rows)
        atomic_write_json(
            cache_path,
            {
                **identity,
                "next_offset": offset,
                "assets": assets,
            },
        )
        print(f"indexed {offset} rows; found {len(assets)}/{len(wanted)} sources")

    missing = sorted(set(wanted) - set(assets))
    if missing:
        raise RuntimeError(f"selected source IDs are absent from Hub dataset: {missing}")
    revision_marker = f"/--/{revision}/--/"
    mismatched = sorted(
        source_id
        for source_id, url in assets.items()
        if revision_marker not in urllib.parse.unquote(url)
    )
    if mismatched:
        raise RuntimeError(
            "datasets-server returned assets from a revision other than the "
            f"pinned commit {revision}: {mismatched}"
        )
    return assets


def _download_validated(url: str, destination: Path, source: SourceClip) -> None:
    """Validate a staged download before atomically replacing its destination."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.qces-v4-",
        suffix=".wav",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        download_file(url, temporary)
        validate_audio(temporary, source)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        partial = temporary.with_suffix(temporary.suffix + ".part")
        if partial.exists():
            partial.unlink()


def _receipt(
    *,
    args: argparse.Namespace,
    metadata_path: Path,
    output_root: Path,
    selected: Sequence[SelectedSource],
) -> Dict[str, Any]:
    return {
        "format": "qces_v4_audiotime_subset_v1",
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "config": args.config,
        "split": args.split,
        "metadata_path": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "selection": {
            "seed": args.seed,
            "sources_per_label": SOURCES_PER_LABEL,
            "semantic_labels": list(SEMANTIC_LABELS),
            "semantic_minimum_interval_seconds": SEMANTIC_MINIMUM_INTERVAL_SECONDS,
            "nuisance_labels": list(NUISANCE_LABELS),
            "nuisance_minimum_interval_seconds": NUISANCE_MINIMUM_INTERVAL_SECONDS,
            "clean_single_event_recordings_only": True,
        },
        "source_count": len(selected),
        "sources": {
            item.source.source_id: {
                "role": item.role,
                "label": item.source.label,
                "interval_seconds": list(item.source.interval_seconds),
                "interval_duration_seconds": item.source.duration_seconds,
                "audio_path": str(
                    (output_root / f"{item.source.source_id}.wav").relative_to(
                        args.audiotime_root.resolve()
                    )
                ),
                "sha256": sha256_file(
                    output_root / f"{item.source.source_id}.wav"
                ),
            }
            for item in selected
        },
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.audiotime_root.resolve()
    args.audiotime_root = root
    metadata_path = root / "timestamp_captions.json"
    if not metadata_path.is_file():
        raise SystemExit(f"missing AudioTime metadata: {metadata_path}")

    selected = selected_sources(root, args.seed)
    output_root = root / "audio"
    output_root.mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in selected
        if args.overwrite
        or not (output_root / f"{item.source.source_id}.wav").is_file()
    ]
    print(
        f"QCES v4 selection: {len(selected)} sources across "
        f"{len(SEMANTIC_LABELS) + len(NUISANCE_LABELS)} labels; "
        f"download: {len(pending)}"
    )

    verify_pinned_revision(args.dataset, args.revision)
    if pending:
        discovery_cache = root / ".qces_v4_asset_discovery.json"
        assets = discover_pinned_assets(
            # Discover the complete stable selection so the cache stays valid
            # even if a later run resumes after only some downloads succeeded.
            [item.source.source_id for item in selected],
            dataset=args.dataset,
            revision=args.revision,
            config=args.config,
            split=args.split,
            page_size=args.page_size,
            cache_path=discovery_cache,
        )
        for index, item in enumerate(pending, start=1):
            destination = output_root / f"{item.source.source_id}.wav"
            _download_validated(
                assets[item.source.source_id], destination, item.source
            )
            print(
                f"downloaded {index}/{len(pending)}: {item.source.source_id} "
                f"({item.role}: {item.source.label})"
            )

    # Validate all selected files, including reused files, before committing the
    # receipt. The receipt is therefore never evidence for a partial acquisition.
    for item in selected:
        path = output_root / f"{item.source.source_id}.wav"
        if not path.is_file():
            raise RuntimeError(f"selected audio file is missing after download: {path}")
        validate_audio(path, item.source)

    receipt = _receipt(
        args=args,
        metadata_path=metadata_path,
        output_root=output_root,
        selected=selected,
    )
    receipt_path = root / "qces_v4_subset_receipt.json"
    atomic_write_json(receipt_path, receipt)
    discovery_cache = root / ".qces_v4_asset_discovery.json"
    if discovery_cache.exists():
        discovery_cache.unlink()
    print(f"QCES v4 AudioTime source pool ready: {output_root}")
    print(f"Acquisition receipt: {receipt_path}")


if __name__ == "__main__":
    main()
