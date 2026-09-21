#!/usr/bin/env python3
"""Create a human-audit queue from clean teacher errors without deleting labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v5_atst_clean_stats_ceiling_v1 import StatsHead


FORMAT = "qces_v5_clean_teacher_class_audit_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1/v5_atst_clean_stats_ceiling_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=base / "atst_clean_stats_ceiling_v1_best.pt")
    parser.add_argument("--cache-dir", type=Path, default=Path("/var/tmp/qces_v5_atst_clean_stats_ceiling_v1"))
    parser.add_argument("--dev-manifest", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5/detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--ontology", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5/ontology_188.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "class_audit_v1")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]


def jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


@torch.inference_mode()
def infer(model: StatsHead, cache: Mapping[str, Any]) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for begin in range(0, len(cache["features"]), 512):
        values.append(model(cache["features"][begin : begin + 512]).cpu())
    return torch.cat(values)


def manifest_events(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scene in read_jsonl(path):
        for event in scene["events"]:
            result.append({"scene_id": scene["scene_id"], **event})
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = StatsHead(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), len(labels))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_data: dict[str, dict[str, Any]] = {}
    for split, manifest in (("dev", args.dev_manifest), ("matched_eval", args.matched_manifest)):
        cache = torch.load(args.cache_dir.resolve() / f"clean_stats_{split}.pt", map_location="cpu", weights_only=False)
        events = manifest_events(manifest)
        if list(cache["event_id"]) != [event["event_id"] for event in events]:
            raise RuntimeError(f"{split} cache/manifest order mismatch")
        scores = infer(model, cache)
        split_data[split] = {"cache": cache, "events": events, "scores": scores}

    rows: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    action_counts = Counter()
    for label_id, label in enumerate(labels):
        item: dict[str, Any] = {"label": label, "label_id": label_id}
        for split in ("dev", "matched_eval"):
            data = split_data[split]
            targets = data["cache"]["targets"].long()
            selected = torch.where(targets == label_id)[0]
            scores = data["scores"][selected]
            predicted = scores.argmax(1)
            accuracy = float(predicted.eq(label_id).float().mean())
            top5 = float((scores.topk(5, dim=1).indices == label_id).any(1).float().mean())
            wrong = Counter(predicted[predicted != label_id].tolist())
            item[split] = {
                "events": int(selected.numel()), "top1_accuracy_↑": accuracy,
                "top5_accuracy_↑": top5,
                "top_confusions": [
                    {"label": labels[index], "count": count}
                    for index, count in wrong.most_common(5)
                ],
            }
        dev = item["dev"]["top1_accuracy_↑"]
        matched = item["matched_eval"]["top1_accuracy_↑"]
        if dev >= 0.60 and matched >= 0.60:
            action = "keep"
        elif dev - matched >= 0.25:
            action = "review_domain_shift"
        elif dev < 0.50 and matched < 0.50:
            action = "review_taxonomy_or_audibility"
        else:
            action = "review_borderline"
        item["action"] = action
        item["dev_minus_matched_top1"] = dev - matched
        rows.append(item)
        action_counts[action] += 1

        if action != "keep":
            data = split_data["matched_eval"]
            targets = data["cache"]["targets"].long()
            selected = torch.where(targets == label_id)[0]
            scores = data["scores"][selected]
            probabilities = scores.softmax(1)
            # Prioritize confident mistakes for human review. This ordering is
            # never used to admit a source into training automatically.
            candidates = []
            for local, global_index in enumerate(selected.tolist()):
                pred = int(scores[local].argmax())
                if pred == label_id:
                    continue
                event = data["events"][global_index]
                candidates.append((
                    float(probabilities[local, pred]),
                    {
                        "label": label,
                        "label_id": label_id,
                        "action": action,
                        "scene_id": event["scene_id"],
                        "event_id": event["event_id"],
                        "component_path": event["component_path"],
                        "source_path": event.get("source_path"),
                        "source_id": event.get("source_id"),
                        "onset_seconds": event["onset_seconds"],
                        "offset_seconds": event["offset_seconds"],
                        "predicted_label": labels[pred],
                        "predicted_confidence": float(probabilities[local, pred]),
                        "gold_confidence": float(probabilities[local, label_id]),
                        "human_decision": None,
                        "human_note": "",
                    },
                ))
            queue.extend(value for _, value in sorted(candidates, key=lambda pair: pair[0], reverse=True)[:3])

    rows.sort(key=lambda row: (row["action"] == "keep", row["matched_eval"]["top1_accuracy_↑"], row["label"]))
    queue.sort(key=lambda row: (row["action"], row["label"], -row["predicted_confidence"]))
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "human-audit routing only; no class or source is automatically removed",
        "classes": len(labels),
        "action_counts": dict(action_counts),
        "queue_items": len(queue),
        "policy": {
            "keep": "clean top1 >=0.60 on both dev and matched official eval",
            "review_domain_shift": "dev minus matched top1 >=0.25",
            "review_taxonomy_or_audibility": "both clean top1 below 0.50",
            "review_borderline": "remaining classes",
            "model_error_used_for_training_selection": False,
            "human_decision_required_before_ontology_change": True,
        },
        "per_class": rows,
        "artifacts": {
            "checkpoint_sha256": _sha256_file(args.checkpoint.resolve()),
            "dev_manifest_sha256": _sha256_file(args.dev_manifest.resolve()),
            "matched_manifest_sha256": _sha256_file(args.matched_manifest.resolve()),
        },
    }
    _atomic_json(report, output_dir / "report.json")
    (output_dir / "manual_listening_queue.jsonl").write_text(jsonl(queue), encoding="utf-8")
    lines = [
        "# Clean semantic class audit",
        "",
        "Không class nào bị xóa tự động. Queue chỉ dùng để nghe và ghi quyết định thủ công.",
        "",
        f"- Classes: {len(labels)}",
        f"- Keep: {action_counts['keep']}",
        f"- Review domain shift: {action_counts['review_domain_shift']}",
        f"- Review taxonomy/audibility: {action_counts['review_taxonomy_or_audibility']}",
        f"- Review borderline: {action_counts['review_borderline']}",
        f"- Listening clips: {len(queue)}",
        "",
        "| Class | Action | Dev top-1 ↑ | Matched top-1 ↑ | Main confusion |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        if row["action"] == "keep":
            continue
        confusion = row["matched_eval"]["top_confusions"]
        main = confusion[0]["label"] if confusion else "—"
        lines.append(
            f"| {row['label']} | {row['action']} | {row['dev']['top1_accuracy_↑']:.3f} | "
            f"{row['matched_eval']['top1_accuracy_↑']:.3f} | {main} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir), "action_counts": dict(action_counts),
        "queue_items": len(queue), "automatic_deletions": 0,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
