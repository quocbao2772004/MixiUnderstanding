#!/usr/bin/env python3
"""Build the final leakage-safe synthetic evidence benchmark for Q-DOR.

The command performs no download.  ``--manifest-only`` validates partition,
scheduling, conditional answerability balance, and output schemas without
requiring materialized audio; omit it for the fully rendered benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.clean_evidence_scenes import (
    DEFAULT_SEED,
    build_clean_evidence_dataset,
    load_source_bank,
)


DEFAULT_SOURCE_BANK = (
    PROJECT_ROOT / "outputs/qces_clean_single_event_source_bank_v1/source_bank.jsonl"
)
DEFAULT_ONTOLOGY = PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/qces_qdor_clean_evidence_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=DEFAULT_SOURCE_BANK)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument(
        "--core-rounds",
        type=int,
        default=0,
        help="0 uses every complete balanced round supported by the rarest class.",
    )
    parser.add_argument("--repeat-rounds", type=int, default=1)
    parser.add_argument("--max-event-seconds", type=float, default=1.20)
    parser.add_argument("--no-distractors", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be in (0, 1)")
    if args.core_rounds < 0 or args.repeat_rounds < 0:
        parser.error("round counts must be non-negative")
    if args.max_event_seconds <= 0:
        parser.error("--max-event-seconds must be positive")
    return args


def load_ontology(path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(labels) != 200 or len(labels) != len(set(labels)):
        raise ValueError(f"final ontology must contain exactly 200 unique labels: {path}")
    return labels


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sources = load_source_bank(
        args.source_bank,
        require_audio_file=not args.manifest_only,
    )
    receipt = build_clean_evidence_dataset(
        sources,
        load_ontology(args.ontology),
        output_dir=args.output_dir,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        core_rounds=None if args.core_rounds == 0 else args.core_rounds,
        repeat_rounds=args.repeat_rounds,
        add_distractors=not args.no_distractors,
        max_event_seconds=args.max_event_seconds,
        render_audio=not args.manifest_only,
        verify_source_hash=args.verify_source_hash,
        overwrite=args.overwrite,
        require_ontology_size=200,
    )
    summary = {
        "format": receipt["format"],
        "passes": receipt["passes"],
        "ontology_size": receipt["ontology_size"],
        "render_audio": receipt["render_audio"],
        "output_dir": str(args.output_dir.resolve()),
        "scenes": {
            split: receipt["schedule"][split]["scenes"]
            for split in ("train", "dev", "test")
        },
        "qa": {
            split: (
                receipt["qa_selection"][split]["selected_answerable"]
                + receipt["qa_selection"][split]["selected_no_evidence"]
            )
            for split in ("train", "dev", "test")
        },
        "shortcut_max_ba": {
            split: max(
                metric["balanced_accuracy"]
                for metric in receipt["metadata_audit"]["qa_text_shortcuts"][split][
                    "baselines"
                ].values()
            )
            for split in ("train", "dev", "test")
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
