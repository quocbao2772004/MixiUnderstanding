#!/usr/bin/env bash
# Run the dependency-light checks used for repository-level validation.

set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}

cd "${repo_root}"
export PYTHONPATH="${repo_root}/code${PYTHONPATH:+:${PYTHONPATH}}"

"${python_bin}" -m compileall -q code/mixi_understanding

tests=(
  test_qces_full200_adaptive_execution.py
  test_qces_real10_schema.py
  test_qces_v5_source_ledger.py
  test_qces_v5_tacos.py
  test_qces_v5_tacos_annotation.py
  test_qces_v5_tacos_finalize.py
  test_qces_v5_upstream_normalizer.py
)

for test_file in "${tests[@]}"; do
  "${python_bin}" -m unittest discover \
    -s code/mixi_understanding/tests \
    -p "${test_file}" \
    -q
done
