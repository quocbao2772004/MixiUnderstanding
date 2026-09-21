#!/usr/bin/env python3
"""Create a 200-class detector initialized from native BEATs-Strong rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from data_util.audioset_classes import as_strong_train_classes  # noqa: E402
from mixi_understanding.qces.supported_ontology import safe_label  # noqa: E402
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import load_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ontology",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt",
    )
    parser.add_argument(
        "--native-checkpoint",
        type=Path,
        default=PRETRAINED_ROOT / "resources/BEATs_strong_1.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_native_supported_detector_200_v1/native_supported_detector.pt",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}; use --overwrite")
    labels = [
        line.strip()
        for line in args.ontology.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 200 or len(set(labels)) != 200:
        raise SystemExit("ontology must contain exactly 200 unique labels")
    native_labels = [safe_label(value) for value in as_strong_train_classes]
    native_index = {label: index for index, label in enumerate(native_labels)}
    missing = sorted(set(labels) - set(native_index))
    if missing:
        raise SystemExit(f"labels absent from native 447 head: {missing}")
    rows = torch.tensor([native_index[label] for label in labels], dtype=torch.long)

    native_state = torch.load(args.native_checkpoint, map_location="cpu", weights_only=True)
    for key in ("strong_head.weight", "strong_head.bias", "weak_head.weight", "weak_head.bias"):
        if key not in native_state or native_state[key].shape[0] != len(native_labels):
            raise RuntimeError(f"unexpected native head tensor {key}: {native_state.get(key, None)}")
    model = load_model(200, "BEATs_strong_1", torch.device("cpu"))
    with torch.no_grad():
        model.strong_head.weight.copy_(native_state["strong_head.weight"].index_select(0, rows))
        model.strong_head.bias.copy_(native_state["strong_head.bias"].index_select(0, rows))
        model.weak_head.weight.copy_(native_state["weak_head.weight"].index_select(0, rows))
        model.weak_head.bias.copy_(native_state["weak_head.bias"].index_select(0, rows))

    payload = {
        "format": "qces_native_supported_detector_200_v1",
        "labels": labels,
        "model_state_dict": model.state_dict(),
        "initialization": {
            "checkpoint": str(args.native_checkpoint.resolve()),
            "checkpoint_sha256": _sha256(args.native_checkpoint.resolve()),
            "native_head_labels": len(native_labels),
            "selected_native_rows": rows.tolist(),
            "ontology": str(args.ontology.resolve()),
            "ontology_sha256": _sha256(args.ontology.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=args.output.parent, prefix=f".{args.output.name}.")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = {
        "format": payload["format"],
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output.resolve()),
        **payload["initialization"],
    }
    receipt_path = args.output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
