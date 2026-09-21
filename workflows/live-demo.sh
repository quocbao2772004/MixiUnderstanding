#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}

export PYTHONPATH="$repo_root/code${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -m streamlit run "$repo_root/live_demo/app.py" \
  --server.port "${PORT:-8515}" \
  --server.address "${ADDRESS:-0.0.0.0}" \
  --server.headless true
