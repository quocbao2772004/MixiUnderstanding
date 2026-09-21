#!/usr/bin/env python3
"""Run a frozen Audio Flamingo 3 auditor on strict QCES-Real-10 renders.

This real-recording evaluator is deliberately separate from the synthetic
AudioQA program: QCES-Real-10 has no oracle waveform stems.  The only acoustic
conditions admitted here are the immutable mixture, predicted evidence and
residual, silence, a question-index-matched shuffled predicted evidence stem,
and a question-only control.  Gold answers are used only after every frozen
auditor call to score its five unmarked option likelihoods.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.real10_scoring import (  # noqa: E402
    ValidatedScoringInputs,
    validate_scoring_inputs,
)
from mixi_understanding.scripts.evaluate_qces_audioqa import (  # noqa: E402
    AudioFlamingo3OptionScorer,
    InputDescriptor,
    _atomic_json,
    _git_provenance,
    _model_source_provenance,
    _package_version,
    _sha256_file,
    _sha256_json,
    append_item,
    bootstrap_paired_metric_summary,
    condition_metrics,
    load_completed_items,
    make_item,
    materialize_input,
    paired_metrics,
    paired_subset_counts,
    score_record_without_gold_inputs,
    validate_metric_metadata,
)


FORMAT_VERSION = "qces_real10_audioqa_audit_v1"
ITEM_FORMAT_VERSION = "qces_real10_audioqa_item_v1"
PROMPT_VERSION = "qces_mc_letter_v1"
SCORING_VERSION = "qces_exact_letter_conditional_logprob_v2"
ALL_CONDITIONS = (
    "mixture",
    "predicted_evidence",
    "predicted_residual",
    "silence",
    "shuffled_evidence",
    "question_only",
)
PREDICTED_CONDITIONS = frozenset(
    {"predicted_evidence", "predicted_residual", "shuffled_evidence"}
)


class Real10AudioQAError(RuntimeError):
    """Raised when a real-data QA-auditor integrity gate fails closed."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("real_dev", "real_test"), default="real_dev"
    )
    parser.add_argument(
        "--allow-real-test",
        action="store_true",
        help="Required only after the renderer's exact method-freeze gate.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=list(ALL_CONDITIONS),
    )
    parser.add_argument(
        "--quantization", choices=("none", "4bit", "8bit"), default="4bit"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hash-model-weights", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    args.conditions = tuple(dict.fromkeys(args.conditions))
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.split == "real_dev" and args.allow_real_test:
        parser.error("--allow-real-test is invalid with --split real_dev")
    local_model = Path(args.model).expanduser().exists()
    if local_model and not args.hash_model_weights:
        parser.error("a local auditor requires --hash-model-weights")
    if not local_model and args.revision is None:
        parser.error("a remote auditor requires an explicit --revision")
    return args


def _safe_file(root: Path, relative: str, context: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise Real10AudioQAError(f"{context} is not a safe POSIX path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise Real10AudioQAError(f"{context} is not a safe relative path")
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise Real10AudioQAError(f"{context} must not be a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise Real10AudioQAError(f"{context} escapes its root") from error
    if not resolved.is_file():
        raise Real10AudioQAError(f"{context} is not a regular file")
    return resolved


def shuffled_record_map(
    validated: ValidatedScoringInputs, *, seed: int
) -> dict[str, str]:
    """Derange within inference-visible question index, never using the answer."""

    groups: dict[tuple[int, str], list[Any]] = defaultdict(list)
    for record in validated.selected_records:
        groups[(record.question_index, record.relation)].append(record)
    result: dict[str, str] = {}
    for (question_index, relation), records in sorted(groups.items()):
        ordered = sorted(
            records,
            key=lambda record: (
                hashlib.sha256(
                    f"qces-real10-shuffle-v1\0{seed}\0{record.sample_id}".encode()
                ).hexdigest(),
                record.sample_id,
            ),
        )
        if len({record.scene_id for record in ordered}) < 2:
            raise Real10AudioQAError(
                "shuffled evidence needs two scenes for "
                f"question_index={question_index}, relation={relation}"
            )
        for index, record in enumerate(ordered):
            source = next(
                ordered[(index + offset) % len(ordered)]
                for offset in range(1, len(ordered) + 1)
                if ordered[(index + offset) % len(ordered)].scene_id != record.scene_id
            )
            result[record.sample_id] = source.sample_id
    return result


def build_input_descriptors(
    validated: ValidatedScoringInputs,
    *,
    conditions: Sequence[str],
    seed: int,
) -> dict[tuple[str, str], InputDescriptor]:
    """Hash every allowed input after strict scorer validation and before AF3."""

    if any(condition not in ALL_CONDITIONS for condition in conditions):
        raise Real10AudioQAError("real evaluator was given a forbidden condition")
    if any("oracle" in condition for condition in conditions):
        raise Real10AudioQAError("QCES-Real-10 cannot consume oracle waveforms")
    predictions = {row["id"]: row for row in validated.prediction_rows}
    records = {record.sample_id: record for record in validated.selected_records}
    shuffled = (
        shuffled_record_map(validated, seed=seed)
        if "shuffled_evidence" in conditions
        else {}
    )

    def file_descriptor(
        *,
        record: Any,
        condition: str,
        source_record_id: str,
        path: Path,
        expected_sha256: str,
    ) -> InputDescriptor:
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise Real10AudioQAError(f"{condition} changed after strict validation")
        return InputDescriptor(
            record_id=record.sample_id,
            condition=condition,
            source_record_id=source_record_id,
            path=str(path),
            sha256=actual_sha256,
            sample_rate=record.sample_rate,
            num_samples=record.num_samples,
        )

    descriptors: dict[tuple[str, str], InputDescriptor] = {}
    for record in validated.selected_records:
        row = predictions[record.sample_id]
        for condition in conditions:
            key = (record.sample_id, condition)
            if condition == "mixture":
                path = _safe_file(
                    validated.scoring_manifest_path.parent,
                    record.mixture_path,
                    f"mixture for {record.sample_id}",
                )
                descriptors[key] = file_descriptor(
                    record=record,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=path,
                    expected_sha256=record.mixture_sha256,
                )
            elif condition in {"predicted_evidence", "predicted_residual"}:
                stem_name = condition.removeprefix("predicted_")
                identity = row["stems"][stem_name]
                path = _safe_file(
                    validated.prediction_root,
                    identity["path"],
                    f"{condition} for {record.sample_id}",
                )
                descriptors[key] = file_descriptor(
                    record=record,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=path,
                    expected_sha256=identity["file_sha256"],
                )
            elif condition == "shuffled_evidence":
                source_id = shuffled[record.sample_id]
                source_identity = predictions[source_id]["stems"]["evidence"]
                path = _safe_file(
                    validated.prediction_root,
                    source_identity["path"],
                    f"shuffled evidence source for {record.sample_id}",
                )
                descriptors[key] = file_descriptor(
                    record=record,
                    condition=condition,
                    source_record_id=source_id,
                    path=path,
                    expected_sha256=source_identity["file_sha256"],
                )
            elif condition == "silence":
                descriptors[key] = InputDescriptor(
                    record_id=record.sample_id,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=None,
                    sha256=hashlib.sha256(
                        f"float32-zero:{record.sample_rate}:{record.num_samples}".encode()
                    ).hexdigest(),
                    sample_rate=record.sample_rate,
                    num_samples=record.num_samples,
                    special="silence",
                )
            elif condition == "question_only":
                descriptors[key] = InputDescriptor(
                    record_id=record.sample_id,
                    condition=condition,
                    source_record_id=record.sample_id,
                    path=None,
                    sha256=hashlib.sha256(b"question-only:no-audio").hexdigest(),
                    sample_rate=record.sample_rate,
                    num_samples=0,
                    special="question_only",
                )
            else:  # pragma: no cover - guarded above
                raise Real10AudioQAError(f"unsupported condition: {condition}")
    if set(descriptors) != {
        (record_id, condition) for record_id in records for condition in conditions
    }:
        raise Real10AudioQAError("descriptor coverage is not exact")
    return descriptors


def materialize_bound_input(
    descriptor: InputDescriptor,
) -> tuple[Any, int | None]:
    """Decode one auditor input while proving its bytes did not change."""

    if descriptor.path is None:
        return materialize_input(descriptor)
    path = Path(descriptor.path)
    before = _sha256_file(path)
    if before != descriptor.sha256:
        raise Real10AudioQAError("auditor input changed before decoding")
    waveform, sample_rate = materialize_input(descriptor)
    after = _sha256_file(path)
    if after != before:
        raise Real10AudioQAError("auditor input changed while decoding")
    return waveform, sample_rate


def build_report(
    *,
    validated: ValidatedScoringInputs,
    conditions: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    run_fingerprint: str,
    metadata: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["condition"])].append(item)
    if set(grouped) != set(conditions):
        raise Real10AudioQAError("completed condition coverage is not exact")
    summaries = {
        condition: condition_metrics(grouped[condition]) for condition in conditions
    }
    paired = paired_metrics(items)
    validate_metric_metadata(summaries, paired)
    intervals, coverage = bootstrap_paired_metric_summary(
        items, bootstrap_samples, seed
    )
    return {
        "format": FORMAT_VERSION,
        "run_fingerprint": run_fingerprint,
        "split": validated.split,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "counts": {
            "records_↑": len(validated.selected_records),
            "unique_scenes_↑": len(
                {record.scene_id for record in validated.selected_records}
            ),
            "unique_creators_↑": len(
                {record.creator_id for record in validated.selected_records}
            ),
            "answerable_records_↑": sum(
                not record.no_evidence for record in validated.selected_records
            ),
            "no_evidence_records_↑": sum(
                record.no_evidence for record in validated.selected_records
            ),
            "conditions_↑": len(conditions),
            "completed_record_conditions_↑": len(items),
        },
        "condition_metrics": summaries,
        "paired_metrics": paired,
        "paired_subset_counts": paired_subset_counts(items),
        "creator_equivalent_scene_cluster_bootstrap_95ci": intervals,
        "creator_equivalent_scene_cluster_bootstrap_coverage": coverage,
        "provenance": metadata,
        "integrity": {
            "strict_real10_scoring_and_render_validation_passed": True,
            "gold_answer_or_index_used_as_auditor_input": False,
            "oracle_or_clean_waveform_conditions_consumed_↓": 0,
            "allowed_conditions_only": sorted(conditions),
            "shuffled_evidence_uses_inference_visible_question_index_only": True,
            "one_unique_creator_per_scene_gate": len(
                {record.creator_id for record in validated.selected_records}
            )
            == len({record.scene_id for record in validated.selected_records}),
            "real_test_method_freeze_gate_verified": (
                True if validated.split == "real_test" else None
            ),
        },
        "interpretation": {
            "sufficiency": (
                "QA(E) on answerable records, conditioned primarily on QA(X) being "
                "correct and question-only being wrong."
            ),
            "necessity": (
                "QA(R) should lose the answer only on answerable records; no residual "
                "leakage endpoint is defined for no-evidence questions."
            ),
            "compactness": (
                "This report does not call evidence minimal; waveform retention is "
                "reported by the separate non-QA scorer."
            ),
        },
        "items_jsonl": "items.jsonl",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validated = validate_scoring_inputs(
        scoring_manifest_path=args.scoring_manifest.resolve(),
        dataset_root=args.dataset_root.resolve(),
        prediction_root=args.prediction_root.resolve(),
        split=args.split,
        allow_real_test=args.allow_real_test,
    )
    descriptors = build_input_descriptors(
        validated,
        conditions=args.conditions,
        seed=args.seed,
    )
    source_inventory = [asdict(descriptors[key]) for key in sorted(descriptors)]
    model_source = _model_source_provenance(args.model, args.hash_model_weights)
    script_path = Path(__file__).resolve()
    run_config = {
        "format": FORMAT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "scoring_version": SCORING_VERSION,
        "script_sha256": _sha256_file(script_path),
        "scoring_manifest_identity": dict(validated.scoring_manifest_identity),
        "prediction_manifest_identity": dict(validated.prediction_manifest_identity),
        "render_receipt_identity": dict(validated.render_receipt_identity),
        "split": args.split,
        "conditions": list(args.conditions),
        "source_inventory_sha256": _sha256_json(source_inventory),
        "model": args.model,
        "revision": args.revision,
        "model_source_inventory_sha256": _sha256_json(model_source),
        "quantization": args.quantization,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "attention_implementation": args.attention_implementation,
        "local_files_only": args.local_files_only,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "runtime_package_versions": {
            name: _package_version(name)
            for name in ("transformers", "torch", "numpy", "scipy", "soundfile")
        },
    }
    run_fingerprint = _sha256_json(run_config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "run_fingerprint": run_fingerprint,
                    "records_↑": len(validated.selected_records),
                    "record_conditions_↑": len(descriptors),
                    "oracle_conditions_consumed_↓": 0,
                },
                indent=2,
            )
        )
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    items_path = output_dir / "items.jsonl"
    report_path = output_dir / "evaluation_report.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != run_fingerprint:
            raise Real10AudioQAError(
                "output belongs to another run; choose a new output directory"
            )
    elif any(output_dir.iterdir()):
        raise Real10AudioQAError("non-empty output lacks bound run metadata")
    else:
        metadata = {
            "run_fingerprint": run_fingerprint,
            "run_config": run_config,
            "source_inventory": source_inventory,
            "git": _git_provenance(),
            "model_source": model_source,
            "runtime_model": None,
            "model_input_contract": {
                "question_text": True,
                "five_unmarked_answer_options": True,
                "condition_audio_or_question_only": True,
                "gold_answer": False,
                "gold_option_index": False,
                "post_hoc_gold_scoring_only": True,
            },
        }
        _atomic_json(metadata_path, metadata)

    completed = load_completed_items(items_path, run_fingerprint)
    expected = set(descriptors)
    if set(completed) - expected:
        raise Real10AudioQAError("items JSONL contains unexpected keys")
    pending = [key for key in sorted(expected) if key not in completed]
    if pending:
        scorer = AudioFlamingo3OptionScorer(
            model_name=args.model,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        metadata = {**metadata, "runtime_model": dict(scorer.provenance())}
        _atomic_json(metadata_path, metadata)
        record_by_id = {
            record.sample_id: record for record in validated.selected_records
        }
        creator_by_id = {
            record.sample_id: record.creator_id for record in validated.selected_records
        }
        for progress, key in enumerate(pending, 1):
            record_id, condition = key
            descriptor = descriptors[key]
            waveform, sample_rate = materialize_bound_input(descriptor)
            record = record_by_id[record_id]
            score = score_record_without_gold_inputs(
                scorer, record, waveform, sample_rate
            )
            item = make_item(record, descriptor, score, run_fingerprint)
            item = {
                **item,
                "format": ITEM_FORMAT_VERSION,
                "creator_id": creator_by_id[record_id],
                "upstream_tacos_split": record.upstream_tacos_split,
            }
            append_item(items_path, item)
            completed[key] = item
            print(
                f"[{progress}/{len(pending)}] {record_id} {condition}: "
                f"{item['predicted_answer']}",
                flush=True,
            )

    ordered_items = [completed[key] for key in sorted(expected)]
    report = build_report(
        validated=validated,
        conditions=args.conditions,
        items=ordered_items,
        run_fingerprint=run_fingerprint,
        metadata=metadata,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _atomic_json(report_path, report)
    print(json.dumps(report["paired_metrics"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
