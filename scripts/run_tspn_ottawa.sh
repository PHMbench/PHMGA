#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_case.sh" \
  --case case1 \
  --config config/tspn_case_exp_ottawa.yaml \
  "$@"             

