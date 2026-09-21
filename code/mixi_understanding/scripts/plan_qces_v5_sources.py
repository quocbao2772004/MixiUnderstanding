#!/usr/bin/env python3
"""Plan (and only when explicitly requested, acquire) QCES v5 sources.

The default command is deliberately metadata-only: it writes a deterministic
source plan and never starts a network download.  ``--download`` is opt-in.
After every planned file exists, ``--finalize-receipt`` hashes the local audio
and commits a receipt that the v5 builder can consume.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.build_qa_removal_dataset import (  # noqa: E402
    SourceClip,
    load_sources,
    sha256_file,
    stable_seed,
)
from mixi_understanding.scripts.download_audiotime_subset import (  # noqa: E402
    DATASET_NAME,
    download_file,
    validate_audio,
)
from mixi_understanding.scripts.download_qces_v4_sources import (  # noqa: E402
    discover_pinned_assets,
    verify_pinned_revision,
)


PLAN_FORMAT = "qces_v5_source_plan_v1"
RECEIPT_FORMAT = "qces_v5_source_receipt_v1"
PINNED_DATASET_REVISION = "50e1a871c03eee734af88ee2b61e13a372a38a8d"
DEFAULT_SEED = 314_159
SPLITS = (
    "train",
    "val",
    "test_iid",
    "test_compositional_ood",
    "test_label_ood",
)


# The smoke inventory is intentionally compatible with the 60-source v4 pool.
# It tests the v5 mechanics; it is not a paper benchmark.
SMOKE_SEEN_LABELS = (
    "Croak",
    "Engine knocking",
    "Jackhammer",
    "Chainsaw",
    "Vacuum cleaner",
    "Fire alarm",
    "Mechanical bell",
)
SMOKE_HELDOUT_LABELS = (
    "Steam whistle",
    "Helicopter",
    "Printer",
    "Wind chime",
    "Rain",
)
SMOKE_NUISANCE_LABELS = (
    "Ambulance (siren)",
    "Sawing",
    "Mechanical fan",
)


# Forty acoustically diverse semantic classes and eight background-like classes
# selected from the pinned 5,000-row AudioTime timestamp metadata.  The planner
# still checks every count instead of trusting this static inventory.
INTERNAL_SCALE_SEEN_LABELS = (
    "Vacuum cleaner",
    "Steam whistle",
    "Mechanical bell",
    "Cheering",
    "Fire alarm",
    "Motorboat, speedboat",
    "Fixed-wing aircraft, airplane",
    "Train",
    "Chainsaw",
    "Crying, sobbing",
    "Helicopter",
    "Shuffling cards",
    "Printer",
    "Rain",
    "Shower",
    "Telephone",
    "Wind chime",
    "Squawk",
    "Thunderstorm",
    "Engine knocking",
    "Jackhammer",
    "Motorcycle",
    "Ducks, geese, waterfowl",
    "Croak",
    "Sewing machine",
    "Toilet flush",
    "Fireworks",
    "Frying (food)",
    "Bicycle bell",
    "Water tap, faucet",
)
INTERNAL_SCALE_HELDOUT_LABELS = (
    "Race car, auto racing",
    "Steam",
    "Trickle, dribble",
    "Lawn mower",
    "Power tool",
    "Stream, river",
    "Frog",
    "Howl",
    "Applause",
    "Fire",
)
INTERNAL_SCALE_NUISANCE_LABELS = (
    "Mechanical fan",
    "Idling",
    "Traffic noise, roadway noise",
    "Music",
    "Rain on surface",
    "Crowd",
    "Waves, surf",
    "Engine",
)


PROFILE = {
    "smoke": {
        "seen_labels": SMOKE_SEEN_LABELS,
        "heldout_labels": SMOKE_HELDOUT_LABELS,
        "nuisance_labels": SMOKE_NUISANCE_LABELS,
        "seen_allocations": {
            "train": 1,
            "val": 1,
            "test_iid": 1,
            "test_compositional_ood": 1,
        },
        "heldout_allocations": {"test_label_ood": 4},
        "nuisance_allocations": {
            "train": 1,
            "val": 1,
            "test_iid": 1,
            "test_compositional_ood": 1,
        },
        "prefer_existing": True,
    },
    # This scale profile is an internal stress test only. AudioTime does not
    # declare a redistributable dataset license, so it is intentionally not
    # named or treated as the paper source route.
    "internal_scale": {
        "seen_labels": INTERNAL_SCALE_SEEN_LABELS,
        "heldout_labels": INTERNAL_SCALE_HELDOUT_LABELS,
        "nuisance_labels": INTERNAL_SCALE_NUISANCE_LABELS,
        # Two recordings per class in every evaluation partition make the
        # repeated-class event challenge source-real rather than a duplicated
        # crop.  Extra training recordings reduce memorization.
        "seen_allocations": {
            "train": 4,
            "val": 2,
            "test_iid": 2,
            "test_compositional_ood": 2,
        },
        "heldout_allocations": {"test_label_ood": 4},
        "nuisance_allocations": {
            "train": 2,
            "val": 1,
            "test_iid": 1,
            "test_compositional_ood": 1,
            "test_label_ood": 1,
        },
        "prefer_existing": False,
    },
}


def _positive_page_size(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be in [1, 100]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE), default="smoke")
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--revision", default=PINNED_DATASET_REVISION)
    parser.add_argument("--config", default="default")
    parser.add_argument("--hub-split", default="train")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--page-size", type=_positive_page_size, default=100)
    parser.add_argument(
        "--plan-path",
        type=Path,
        help="default: <AudioTime root>/qces_v5_<profile>_source_plan.json",
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help="default: <AudioTime root>/qces_v5_<profile>_source_receipt.json",
    )
    parser.add_argument(
        "--license-evidence-json",
        type=Path,
        help=(
            "Optional reviewed license record. Without it, redistribution is "
            "conservatively marked unverified/false."
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Opt in to downloading every missing source in the plan.",
    )
    parser.add_argument(
        "--finalize-receipt",
        action="store_true",
        help="Require all planned files, hash them, and write the final receipt.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _default_license_record() -> Dict[str, Any]:
    # Neither the pinned Hugging Face card metadata nor the upstream GitHub
    # repository declares a dataset license.  Do not infer permission from
    # public downloadability.
    return {
        "record_id": "audiotime-license-unverified-2026-07-21",
        "status": "unverified",
        "declared_license": None,
        "redistribution_allowed": False,
        "commercial_use_allowed": None,
        "derivatives_allowed": None,
        "checked_on": "2026-07-21",
        "evidence_urls": [
            "https://github.com/zeyuxie29/AudioTime",
            "https://huggingface.co/datasets/enyoukai/audiotime-timestamps",
        ],
        "observation": (
            "The pinned Hugging Face card has no license field and the upstream "
            "repository exposes no LICENSE file; author confirmation is required "
            "before redistributing audio or derived stems."
        ),
    }


def _license_record(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return _default_license_record()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "record_id",
        "status",
        "declared_license",
        "redistribution_allowed",
        "commercial_use_allowed",
        "derivatives_allowed",
        "checked_on",
        "evidence_urls",
        "observation",
    }
    if set(payload) != required:
        raise ValueError(
            "license evidence fields mismatch: "
            f"missing={sorted(required-set(payload))}, "
            f"extra={sorted(set(payload)-required)}"
        )
    if payload["status"] not in {"verified", "unverified", "restricted"}:
        raise ValueError("unsupported license evidence status")
    if not isinstance(payload["redistribution_allowed"], bool):
        raise TypeError("redistribution_allowed must be boolean")
    if payload["redistribution_allowed"] and payload["status"] != "verified":
        raise ValueError("redistribution requires verified license evidence")
    return payload


def _eligible_by_label(
    sources: Iterable[SourceClip], minimum_duration: float
) -> Dict[str, List[SourceClip]]:
    grouped: Dict[str, List[SourceClip]] = defaultdict(list)
    for source in sources:
        if source.duration_seconds + 1e-9 >= minimum_duration:
            grouped[source.label].append(source)
    return grouped


def _ordered_sources(
    sources: Sequence[SourceClip], *, seed: int, role: str, label: str,
    prefer_existing: bool,
) -> List[SourceClip]:
    return sorted(
        sources,
        key=lambda source: (
            0 if prefer_existing and source.audio_path.is_file() else 1,
            stable_seed(seed, "qces-v5-source", role, label, source.source_id),
            source.source_id,
        ),
    )


def _select_role(
    *, grouped: Mapping[str, Sequence[SourceClip]], labels: Sequence[str],
    allocations: Mapping[str, int], role: str, minimum_duration: float,
    seed: int, prefer_existing: bool,
) -> List[Tuple[str, str, SourceClip]]:
    needed = sum(allocations.values())
    selected: List[Tuple[str, str, SourceClip]] = []
    for label in labels:
        candidates = _ordered_sources(
            grouped.get(label, ()), seed=seed, role=role, label=label,
            prefer_existing=prefer_existing,
        )
        if len(candidates) < needed:
            raise RuntimeError(
                f"{role} label {label!r} has {len(candidates)} eligible "
                f">={minimum_duration:.1f}s sources; need {needed}"
            )
        cursor = 0
        for partition, count in allocations.items():
            for source in candidates[cursor : cursor + count]:
                selected.append((role, partition, source))
            cursor += count
    return selected


def make_plan(
    *, audiotime_root: Path, profile: str, seed: int, dataset: str,
    revision: str, config: str, hub_split: str,
    license_record: Mapping[str, Any],
) -> Dict[str, Any]:
    design = PROFILE[profile]
    sources = load_sources(audiotime_root, require_audio=False)
    semantic_grouped = _eligible_by_label(sources, 1.5)
    nuisance_grouped = _eligible_by_label(sources, 5.8)
    selected = []
    selected.extend(
        _select_role(
            grouped=semantic_grouped,
            labels=design["seen_labels"],
            allocations=design["seen_allocations"],
            role="semantic_seen",
            minimum_duration=1.5,
            seed=seed,
            prefer_existing=bool(design["prefer_existing"]),
        )
    )
    selected.extend(
        _select_role(
            grouped=semantic_grouped,
            labels=design["heldout_labels"],
            allocations=design["heldout_allocations"],
            role="semantic_heldout",
            minimum_duration=1.5,
            seed=seed,
            prefer_existing=bool(design["prefer_existing"]),
        )
    )
    selected.extend(
        _select_role(
            grouped=nuisance_grouped,
            labels=design["nuisance_labels"],
            allocations=design["nuisance_allocations"],
            role="nuisance",
            minimum_duration=5.8,
            seed=seed,
            prefer_existing=bool(design["prefer_existing"]),
        )
    )
    source_ids = [source.source_id for _, _, source in selected]
    if len(source_ids) != len(set(source_ids)):
        raise AssertionError("a source recording was assigned more than one partition")
    metadata_path = audiotime_root / "timestamp_captions.json"
    rows = []
    for role, partition, source in selected:
        local = source.audio_path
        rows.append(
            {
                "source_id": source.source_id,
                "label": source.label,
                "role": role,
                "partition": partition,
                "source_dataset": "AudioTime",
                "dataset_version": revision,
                "creator_id": "unknown",
                "uploader_id": "unknown",
                "attribution": "AudioTime upstream; original creator metadata unavailable",
                "source_license_spdx": "NOASSERTION",
                "source_license_url": "https://github.com/zeyuxie29/AudioTime",
                "source_interval_seconds": list(source.interval_seconds),
                "interval_duration_seconds": source.duration_seconds,
                "audio_path": _relative(local, PROJECT_ROOT),
                "acquired": local.is_file(),
                "sha256": sha256_file(local) if local.is_file() else None,
                "license_record_id": license_record["record_id"],
            }
        )
    counts_by_partition: Dict[str, int] = defaultdict(int)
    for row in rows:
        counts_by_partition[row["partition"]] += 1
    return {
        "format": PLAN_FORMAT,
        "profile": profile,
        "seed": seed,
        "dataset": dataset,
        "dataset_revision": revision,
        "config": config,
        "hub_split": hub_split,
        "metadata_path": _relative(metadata_path, PROJECT_ROOT),
        "metadata_sha256": sha256_file(metadata_path),
        "selection": {
            "clean_single_event_recordings_only": True,
            "semantic_minimum_interval_seconds": 1.5,
            "nuisance_minimum_interval_seconds": 5.8,
            "seen_labels": list(design["seen_labels"]),
            "heldout_labels": list(design["heldout_labels"]),
            "nuisance_labels": list(design["nuisance_labels"]),
            "seen_allocations": design["seen_allocations"],
            "heldout_allocations": design["heldout_allocations"],
            "nuisance_allocations": design["nuisance_allocations"],
        },
        "source_count": len(rows),
        "source_counts_by_partition": dict(sorted(counts_by_partition.items())),
        "acquisition_complete": all(row["acquired"] for row in rows),
        "license_record": dict(license_record),
        "sources": sorted(rows, key=lambda row: row["source_id"]),
        "planner_identity": {
            "path": _relative(Path(__file__), PROJECT_ROOT),
            "sha256": sha256_file(Path(__file__)),
            "python_version": platform.python_version(),
        },
    }


def _source_lookup(audiotime_root: Path) -> Dict[str, SourceClip]:
    return {
        source.source_id: source
        for source in load_sources(audiotime_root, require_audio=False)
    }


def acquire_missing(
    plan: Mapping[str, Any], *, audiotime_root: Path, page_size: int,
) -> None:
    missing = [row for row in plan["sources"] if not Path(PROJECT_ROOT / row["audio_path"]).is_file()]
    if not missing:
        return
    verify_pinned_revision(plan["dataset"], plan["dataset_revision"])
    assets = discover_pinned_assets(
        [row["source_id"] for row in missing],
        dataset=plan["dataset"],
        revision=plan["dataset_revision"],
        config=plan["config"],
        split=plan["hub_split"],
        page_size=page_size,
        cache_path=audiotime_root / ".qces_v5_asset_discovery.json",
    )
    metadata = _source_lookup(audiotime_root)
    for index, row in enumerate(missing, start=1):
        source = metadata[row["source_id"]]
        destination = PROJECT_ROOT / row["audio_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        download_file(assets[source.source_id], destination)
        validate_audio(destination, source)
        print(f"downloaded {index}/{len(missing)}: {source.source_id}")


def finalize_receipt(plan: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for row in plan["sources"]:
        path = PROJECT_ROOT / row["audio_path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"planned source is missing: {path}; run with --download first"
            )
        committed = dict(row)
        committed["acquired"] = True
        committed["sha256"] = sha256_file(path)
        rows.append(committed)
    receipt = dict(plan)
    receipt["format"] = RECEIPT_FORMAT
    receipt["acquisition_complete"] = True
    receipt["sources"] = rows
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.audiotime_root.resolve()
    args.plan_path = args.plan_path or root / f"qces_v5_{args.profile}_source_plan.json"
    args.receipt_path = args.receipt_path or root / f"qces_v5_{args.profile}_source_receipt.json"
    license_record = _license_record(args.license_evidence_json)
    plan = make_plan(
        audiotime_root=root,
        profile=args.profile,
        seed=args.seed,
        dataset=args.dataset,
        revision=args.revision,
        config=args.config,
        hub_split=args.hub_split,
        license_record=license_record,
    )
    _atomic_json(args.plan_path, plan, args.overwrite)
    missing = sum(not row["acquired"] for row in plan["sources"])
    print(
        f"source plan: {args.plan_path} ({plan['source_count']} sources; "
        f"{missing} missing; no download={not args.download})"
    )
    if args.download:
        acquire_missing(plan, audiotime_root=root, page_size=args.page_size)
        # Recompute local hashes after acquisition.
        plan = make_plan(
            audiotime_root=root,
            profile=args.profile,
            seed=args.seed,
            dataset=args.dataset,
            revision=args.revision,
            config=args.config,
            hub_split=args.hub_split,
            license_record=license_record,
        )
        _atomic_json(args.plan_path, plan, True)
    if args.finalize_receipt:
        receipt = finalize_receipt(plan)
        _atomic_json(args.receipt_path, receipt, args.overwrite)
        print(f"source receipt: {args.receipt_path}")


if __name__ == "__main__":
    main()
