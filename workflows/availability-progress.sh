#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${QCES_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
STATE_PATH="${1:-${PROJECT_ROOT}/outputs/qces_audioset_official_strong_availability_v1/availability_state.json}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to read the availability state" >&2
  exit 1
fi

if [[ ! -f "${STATE_PATH}" ]]; then
  echo "State file not found: ${STATE_PATH}" >&2
  exit 1
fi

jq '{
  done: (.fragment_receipts | length),
  total: ([.shard_metadata[].num_row_groups] | add),
  percent: (((.fragment_receipts | length) * 10000 / ([.shard_metadata[].num_row_groups] | add)) | round / 100),
  matches: ([.fragment_receipts[].indexed_rows] | add),
  updated_unix_seconds: .updated_unix_seconds
}' "${STATE_PATH}"
