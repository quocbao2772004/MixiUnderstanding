#!/usr/bin/env bash
# Materialize and acoustically grade the frozen 200-class primary contract.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${QCES_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${QCES_PYTHON_BIN:-python3}"
EXECUTION_DIR="${QCES_FULL200_DIR:-${PROJECT_ROOT}/outputs/qces_full200_adaptive_v1}"
SCRATCH_ROOT="${QCES_FULL200_SCRATCH_ROOT:-${TMPDIR:-/tmp}/qces_full200_adaptive_v1}"
MODE="${1:-all}"
EXISTING_AUDIOSET_TRAIN="${QCES_EXISTING_AUDIOSET_TRAIN:-${PROJECT_ROOT}/outputs/qces_audioset_strong_subset_qces200_q40_10/audioset_strong_detector_manifest_train.partial.jsonl}"
EXISTING_AUDIOSET_EVAL="${QCES_EXISTING_AUDIOSET_EVAL:-${PROJECT_ROOT}/outputs/qces_audioset_strong_subset_qces200_q40_10/audioset_strong_detector_manifest_test.partial.jsonl}"

export PYTHONPATH="${PROJECT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

if [[ ! -f "${EXECUTION_DIR}/execution_contract.json" ]]; then
  echo "Missing execution contract: ${EXECUTION_DIR}/execution_contract.json" >&2
  exit 1
fi
if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" ]]; then
  echo "Usage: $0 [train|eval|all]" >&2
  exit 2
fi

mkdir -p "${SCRATCH_ROOT}"
touch "${SCRATCH_ROOT}/.qces_full200_task_owned_scratch"

mapfile -t CHUNK_CONTRACTS < <(
  "${PYTHON_BIN}" - "${EXECUTION_DIR}/primary_chunk_registry.jsonl" "${MODE}" <<'PY'
import json
import sys

path, mode = sys.argv[1:]
order = {"train": 0, "eval": 1}
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
rows.sort(key=lambda row: (order[row["metadata_split"]], row["chunk_id"]))
for row in rows:
    if mode != "all" and row["metadata_split"] != mode:
        continue
    fields = [
        row["chunk_id"],
        row["metadata_split"],
        row["source_route"],
        row["plan_path"],
        row["plan_sha256"],
        row["hf_dataset"],
        row["hf_revision"],
        str(row["source_videos"]),
        str(row["crop_requests"]),
    ]
    if any("\t" in str(value) or "\n" in str(value) for value in fields):
        raise SystemExit("unsafe tab/newline in chunk contract")
    print("\t".join(fields))
PY
)

process_chunk() {
  local contract="$1"
  local chunk_id split route plan plan_sha dataset revision source_videos crop_requests
  IFS=$'\t' read -r chunk_id split route plan plan_sha dataset revision source_videos crop_requests <<<"${contract}"
  local clean_dir="${EXECUTION_DIR}/primary_clean/${split}/${chunk_id}"
  local terminal_receipt="${clean_dir}/source_bank_receipt.json"
  if [[ -f "${terminal_receipt}" ]]; then
    if "${PYTHON_BIN}" - "${terminal_receipt}" "${crop_requests}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
ok = (
    int(receipt.get("input_items", -1)) == expected
    and int(receipt.get("accepted_items", -1))
      + int(receipt.get("rejected_items", -1)) == expected
)
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[skip] ${chunk_id}: terminal receipt already valid"
      return
    fi
    echo "Invalid existing terminal receipt: ${terminal_receipt}" >&2
    exit 1
  fi

  local materialized_dir
  if [[ "${split}" == "train" ]]; then
    materialized_dir="${SCRATCH_ROOT}/${chunk_id}"
  else
    materialized_dir="${EXECUTION_DIR}/fixed_eval_materialized/${chunk_id}"
  fi
  mkdir -p "${materialized_dir}"
  touch "${materialized_dir}/.qces_full200_chunk"

  actual_plan_sha="$(sha256sum "${plan}" | awk '{print $1}')"
  if [[ "${actual_plan_sha}" != "${plan_sha}" ]]; then
    echo "Plan hash mismatch for ${chunk_id}" >&2
    exit 1
  fi

  common_args=(
    --plan "${plan}"
    --output-dir "${materialized_dir}"
    --hf-dataset "${dataset}"
    --resolved-revision "${revision}"
    --storage-mode requested_crops
    --require-preindexed-locations
    --existing-manifest "${EXISTING_AUDIOSET_TRAIN}"
    --existing-manifest "${EXISTING_AUDIOSET_EVAL}"
    --min-free-disk-gib 10
    --scan-workers 4
    --max-retries 10
    --retry-initial-seconds 2
    --retry-max-seconds 30
  )

  # The agkphysics Parquet layout stores 100 encoded clips in each audio
  # column chunk.  Fetching sparse rows via Parquet wastes roughly 100x I/O.
  # Resolve the exact same pinned FLAC through the HF rows API, then let the
  # existing transaction-safe crop materializer consume it as a source
  # manifest.  Completed crop fragments are excluded and remain resumable.
  if [[ "${dataset}" == "agkphysics/AudioSet" ]]; then
    local direct_dir="${materialized_dir}/direct_asset_sources"
    echo "[direct-assets] ${chunk_id}: workers=12"
    "${PYTHON_BIN}" \
      "${PROJECT_ROOT}/code/mixi_understanding/scripts/fetch_qces_agkphysics_direct_assets.py" \
      --plan "${plan}" \
      --output-dir "${direct_dir}" \
      --materialized-dir "${materialized_dir}" \
      --shard-index "${PROJECT_ROOT}/outputs/qces_audioset_official_strong_availability_v1/audioset_availability_shards.partial.jsonl" \
      --workers 12 \
      --resolve-workers 2 \
      --attempts 20
    common_args+=(
      --existing-manifest "${direct_dir}/direct_asset_source_manifest.jsonl"
    )
  fi

  echo "[scan] ${chunk_id}: videos=${source_videos} crops=${crop_requests} route=${route}"
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/materialize_qces_supported_audioset_plan.py" \
    "${common_args[@]}" \
    --scan-only

  echo "[materialize] ${chunk_id}"
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/materialize_qces_supported_audioset_plan.py" \
    "${common_args[@]}" \
    --max-new-videos "${source_videos}"

  echo "[acoustic-gate] ${chunk_id}"
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/build_qces_audiosep_clean_source_bank.py" \
    --manifest "${materialized_dir}/audioset_strong_crop_manifest.jsonl" \
    --output-dir "${clean_dir}" \
    --device cuda \
    --batch-size 1 \
    --shard-size 256 \
    --max-items "${crop_requests}" \
    --target-train-per-class 100 \
    --target-eval-per-class 20

  "${PYTHON_BIN}" - "${terminal_receipt}" "${crop_requests}" <<'PY'
import json
import sys
receipt = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
assert int(receipt["input_items"]) == expected
assert int(receipt["accepted_items"]) + int(receipt["rejected_items"]) == expected
assert receipt["invariants"]["qa_answers_used_for_selection"] is False
assert receipt["invariants"]["downstream_accuracy_used_for_selection"] is False
assert receipt["invariants"]["same_fixed_quality_thresholds_for_all_splits"] is True
PY

  mkdir -p "${clean_dir}/transport_archive"
  cp "${materialized_dir}/materialization_receipt.json" "${clean_dir}/transport_archive/"
  cp "${materialized_dir}/audioset_strong_crop_manifest.jsonl" "${clean_dir}/transport_archive/"
  cp "${materialized_dir}/audioset_strong_plan_manifest.jsonl" "${clean_dir}/transport_archive/"

  if [[ "${split}" == "train" ]]; then
    "${PYTHON_BIN}" - "${SCRATCH_ROOT}" "${materialized_dir}" <<'PY'
import pathlib
import shutil
import sys
root = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2]).resolve()
if not (root / ".qces_full200_task_owned_scratch").is_file():
    raise SystemExit("scratch root marker is missing")
if target.parent != root or not (target / ".qces_full200_chunk").is_file():
    raise SystemExit(f"refusing cleanup outside exact marked chunk: {target}")
shutil.rmtree(target)
PY
    echo "[cleanup] removed task-owned train scratch ${materialized_dir}"
  fi

  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/report_qces_full200_progress.py" \
    --execution-dir "${EXECUTION_DIR}" >/dev/null
}

for contract in "${CHUNK_CONTRACTS[@]}"; do
  process_chunk "${contract}"
done

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/code/mixi_understanding/scripts/report_qces_full200_progress.py" \
  --execution-dir "${EXECUTION_DIR}"

if [[ "${MODE}" == "all" ]]; then
  QUALITY_ARGS=()
  while IFS= read -r quality_path; do
    QUALITY_ARGS+=(--quality-audit "${quality_path}")
  done < <(
    find "${EXECUTION_DIR}/primary_clean" -type f -name quality_audit.jsonl -print | sort
  )
  set +e
  "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/build_qces_reserve_quality_topup.py" \
    --primary-plan "${PROJECT_ROOT}/outputs/qces_availability_aware_acoustic_plan_v2/crop_source_plan_train.jsonl" \
    --primary-plan "${PROJECT_ROOT}/outputs/qces_availability_aware_acoustic_plan_v2/crop_source_plan_eval.jsonl" \
    "${QUALITY_ARGS[@]}" \
    --target-train-per-class 100 \
    --target-eval-per-class 20 \
    --overdraw-factor 2 \
    --output-dir "${EXECUTION_DIR}/topup_round1"
  topup_status=$?
  set -e
  if [[ "${topup_status}" -ne 0 && "${topup_status}" -ne 2 ]]; then
    exit "${topup_status}"
  fi
fi
