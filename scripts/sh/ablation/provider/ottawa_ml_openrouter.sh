#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../_common.sh"

phmga_enter_repo "$SCRIPT_DIR"
# Provider qualification candidate wrapper.
phmga_prepare_fixed_provider "openrouter" "stepfun/step-3.5-flash:free"

OUTPUT_DIR="${PHMGA_OUTPUT_DIR:-artifacts/paper/ottawa_ml_openrouter_v1}"

echo "Running Ottawa ML provider qualification candidate..."
phmga_log_provider_choice
python main.py +runs=ottawa_ml "${PHMGA_PROVIDER_ARGS[@]}" runtime.output_dir="$OUTPUT_DIR"
phmga_assert_file "$OUTPUT_DIR/final_report.md"
echo "Completed provider qualification candidate: $OUTPUT_DIR"
echo "Next: record the provider qualification result in doc/experiments/01_result_ledger.md"
