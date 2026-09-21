#!/usr/bin/env python3
"""Build a pinned, resumable physical-row index for AudioSet HF mirrors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

from mixi_understanding.qces.audioset_availability_index import (
    DEFAULT_HF_DATASET,
    DEFAULT_PARQUET_PATTERN,
    DEFAULT_SOURCE_ROUTE,
    AvailabilityIndexConfig,
    AvailabilityIndexError,
    AvailabilityAllowlist,
    AvailabilityRoute,
    RetryPolicy,
    run_availability_index,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_audioset_availability_index_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--route-json",
        action="append",
        default=[],
        help=(
            "Repeatable JSON route object with source_route, hf_dataset, "
            "parquet_pattern, splits and hf_revision or resolved_revision."
        ),
    )
    parser.add_argument(
        "--route-file",
        type=Path,
        action="append",
        default=[],
        help="Repeatable JSON file containing one route object or a list of routes.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        action="append",
        default=[],
        help="Global video-ID allowlist; repeatable files are unioned by scope.",
    )
    parser.add_argument(
        "--allowlist-json",
        action="append",
        default=[],
        help=(
            "Repeatable scoped JSON object: path, source_route ('*' allowed), "
            "hf_split ('*' allowed)."
        ),
    )
    parser.add_argument("--source-route", default=DEFAULT_SOURCE_ROUTE)
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--resolved-revision", default="")
    parser.add_argument("--parquet-pattern", default=DEFAULT_PARQUET_PATTERN)
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Default-route logical split; repeatable. Defaults to train/test.",
    )
    parser.add_argument("--scan-workers", type=int, default=4)
    parser.add_argument("--block-size-mib", type=int, default=1)
    parser.add_argument("--max-new-row-groups", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-initial-seconds", type=float, default=0.5)
    parser.add_argument("--retry-max-seconds", type=float, default=8.0)
    parser.add_argument(
        "--require-unique-video-ids",
        action="store_true",
        help=(
            "Fail publication when a video_id has multiple physical locations. "
            "By default every location is retained and duplicates are reported."
        ),
    )
    args = parser.parse_args(argv)
    if args.scan_workers < 1:
        parser.error("--scan-workers must be >= 1")
    if args.block_size_mib < 1:
        parser.error("--block-size-mib must be >= 1")
    if args.max_new_row_groups < 0:
        parser.error("--max-new-row-groups must be >= 0")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.retry_initial_seconds < 0 or args.retry_max_seconds < 0:
        parser.error("retry delays must be non-negative")
    return args


def _load_route_file(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AvailabilityIndexError(f"cannot read route file {path}: {error}") from error
    values = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(value, dict) for value in values):
        raise AvailabilityIndexError(f"route file must contain object(s): {path}")
    return [dict(value) for value in values]


def _route_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for raw in args.route_json:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AvailabilityIndexError(f"invalid --route-json: {error}") from error
        if not isinstance(payload, dict):
            raise AvailabilityIndexError("--route-json must be one JSON object")
        values.append(payload)
    for path in args.route_file:
        values.extend(_load_route_file(path))
    if values:
        return values
    return [
        {
            "source_route": args.source_route,
            "hf_dataset": args.hf_dataset,
            "hf_revision": args.hf_revision,
            "resolved_revision": args.resolved_revision,
            "parquet_pattern": args.parquet_pattern,
            "splits": args.split or ["train", "test"],
            "optional_metadata_columns": ["labels", "human_labels"],
        }
    ]


def _resolve_revision(
    spec: Mapping[str, Any],
    *,
    retry: RetryPolicy,
) -> str:
    explicit = str(spec.get("resolved_revision") or "").strip()
    if explicit:
        return explicit
    pattern = str(spec.get("parquet_pattern") or "")
    if not pattern.startswith("hf://"):
        return "local-unpinned"
    dataset = str(spec.get("hf_dataset") or "")
    revision = str(spec.get("hf_revision") or "main")
    last_error: BaseException | None = None
    for attempt in range(1, retry.max_attempts + 1):
        try:
            info = HfApi().dataset_info(dataset, revision=revision)
            if not info.sha:
                raise AvailabilityIndexError(
                    f"could not resolve immutable revision for {dataset}@{revision}"
                )
            return str(info.sha)
        except (OSError, TimeoutError, HfHubHTTPError) as error:
            last_error = error
            if attempt >= retry.max_attempts:
                break
            delay = min(
                retry.maximum_delay_seconds,
                retry.initial_delay_seconds * (2 ** (attempt - 1)),
            )
            print(
                f"retry {attempt}/{retry.max_attempts - 1} resolving "
                f"{dataset}@{revision} after {type(error).__name__}: {error}; "
                f"sleeping {delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AvailabilityIndexError(
        f"failed resolving {dataset}@{revision}: {last_error}"
    ) from last_error


def _build_routes(
    args: argparse.Namespace,
    *,
    retry: RetryPolicy,
) -> tuple[AvailabilityRoute, ...]:
    routes: list[AvailabilityRoute] = []
    for spec in _route_specs(args):
        source_route = str(spec.get("source_route") or "").strip()
        dataset = str(spec.get("hf_dataset") or "").strip()
        pattern = str(spec.get("parquet_pattern") or "").strip()
        splits_value = spec.get("splits")
        if not isinstance(splits_value, list):
            raise AvailabilityIndexError(
                f"route {source_route!r} must provide splits as a JSON list"
            )
        metadata_value = spec.get(
            "optional_metadata_columns", ["labels", "human_labels"]
        )
        if not isinstance(metadata_value, list):
            raise AvailabilityIndexError(
                f"route {source_route!r} optional_metadata_columns must be a list"
            )
        routes.append(
            AvailabilityRoute(
                source_route=source_route,
                hf_dataset=dataset,
                resolved_revision=_resolve_revision(spec, retry=retry),
                parquet_pattern=pattern,
                splits=tuple(str(value) for value in splits_value),
                optional_metadata_columns=tuple(
                    str(value) for value in metadata_value
                ),
            )
        )
    return tuple(routes)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_allowlist_ids(path: Path) -> tuple[frozenset[str], int]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AvailabilityIndexError(f"allowlist does not exist: {resolved}")
    if resolved.suffix.lower() == ".json":
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise AvailabilityIndexError(
                f"invalid allowlist JSON {resolved}: {error}"
            ) from error
        values = payload.get("video_ids", []) if isinstance(payload, dict) else payload
        if not isinstance(values, list):
            raise AvailabilityIndexError(
                f"allowlist JSON must be a list or contain video_ids: {resolved}"
            )
        rows = [str(value).strip() for value in values if str(value).strip()]
        return frozenset(rows), len(rows)

    rows: list[str] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            if value.startswith("{"):
                try:
                    row = json.loads(value)
                except json.JSONDecodeError as error:
                    raise AvailabilityIndexError(
                        f"invalid allowlist JSON at {resolved}:{line_number}: {error}"
                    ) from error
                video_id = str(
                    row.get("video_id") or row.get("source_video_id") or ""
                ).strip()
            else:
                video_id = value.split("\t", 1)[0].split(",", 1)[0].strip()
                if line_number == 1 and video_id.lower() in {
                    "video_id",
                    "source_video_id",
                }:
                    continue
            if not video_id:
                raise AvailabilityIndexError(
                    f"empty video_id at {resolved}:{line_number}"
                )
            rows.append(video_id)
    if not rows:
        raise AvailabilityIndexError(f"empty allowlist: {resolved}")
    return frozenset(rows), len(rows)


def _build_allowlists(args: argparse.Namespace) -> tuple[AvailabilityAllowlist, ...]:
    specs: list[dict[str, Any]] = [
        {"path": str(path), "source_route": "*", "hf_split": "*"}
        for path in args.allowlist
    ]
    for raw in args.allowlist_json:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AvailabilityIndexError(f"invalid --allowlist-json: {error}") from error
        if not isinstance(payload, dict):
            raise AvailabilityIndexError("--allowlist-json must be one JSON object")
        specs.append(payload)
    grouped: dict[tuple[str, str], list[Path]] = {}
    for spec in specs:
        scope = (
            str(spec.get("source_route") or "*"),
            str(spec.get("hf_split") or "*"),
        )
        path_value = str(spec.get("path") or "")
        if not path_value:
            raise AvailabilityIndexError(f"allowlist scope {scope} has no path")
        grouped.setdefault(scope, []).append(Path(path_value).resolve())

    output: list[AvailabilityAllowlist] = []
    for (source_route, hf_split), paths in sorted(grouped.items()):
        video_ids: set[str] = set()
        input_rows = 0
        source_hashes: list[dict[str, str]] = []
        for path in sorted(paths):
            values, rows = _load_allowlist_ids(path)
            video_ids.update(values)
            input_rows += rows
            source_hashes.append(
                {"path": str(path), "sha256": _sha256_file(path)}
            )
        normalized = hashlib.sha256(
            json.dumps(
                sorted(video_ids), separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        sources_hash = hashlib.sha256(
            json.dumps(
                source_hashes,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        output.append(
            AvailabilityAllowlist(
                video_ids=frozenset(video_ids),
                normalized_sha256=normalized,
                source_path=";".join(str(path) for path in sorted(paths)),
                source_sha256=sources_hash,
                input_rows=input_rows,
                source_route=source_route,
                hf_split=hf_split,
            )
        )
    return tuple(output)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    retry = RetryPolicy(
        max_attempts=int(args.max_retries),
        initial_delay_seconds=float(args.retry_initial_seconds),
        maximum_delay_seconds=float(args.retry_max_seconds),
    )
    try:
        config = AvailabilityIndexConfig(
            routes=_build_routes(args, retry=retry),
            output_dir=args.output_dir,
            allowlists=_build_allowlists(args),
            scan_workers=int(args.scan_workers),
            block_size_bytes=int(args.block_size_mib) * (1 << 20),
            max_new_row_groups=int(args.max_new_row_groups),
            require_unique_video_ids=bool(args.require_unique_video_ids),
            retry=retry,
        )
        receipt = run_availability_index(config)
    except (AvailabilityIndexError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if bool(receipt["complete"] and receipt["integrity_pass"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
