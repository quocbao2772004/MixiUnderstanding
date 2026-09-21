#!/usr/bin/env bash
# Supervise the resumable full-200 materializer across transient network faults.

set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="${QCES_PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
MAX_RUN_ATTEMPTS="${QCES_SUPERVISOR_MAX_ATTEMPTS:-30}"
RESTART_DELAY_SECONDS="${QCES_SUPERVISOR_RESTART_DELAY_SECONDS:-60}"
export PYTHONPATH="${PROJECT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

for ((attempt = 1; attempt <= MAX_RUN_ATTEMPTS; attempt++)); do
  echo "[supervisor] run ${attempt}/${MAX_RUN_ATTEMPTS} started $(date --iso-8601=seconds)"
  bash "${PROJECT_ROOT}/workflows/full200-primary.sh" all
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    echo "[supervisor] complete $(date --iso-8601=seconds)"
    exit 0
  fi
  echo "[supervisor] run ${attempt} exited ${status}; durable chunks will be skipped on resume"
  "${QCES_PYTHON_BIN:-python3}" \
    "${PROJECT_ROOT}/code/mixi_understanding/scripts/report_qces_full200_progress.py" \
    --execution-dir "${QCES_FULL200_DIR:-${PROJECT_ROOT}/outputs/qces_full200_adaptive_v1}" || true
  if ((attempt == MAX_RUN_ATTEMPTS)); then
    break
  fi
  echo "[supervisor] retrying in ${RESTART_DELAY_SECONDS}s"
  sleep "${RESTART_DELAY_SECONDS}"
done

echo "[supervisor] exhausted ${MAX_RUN_ATTEMPTS} runs" >&2
exit 1
