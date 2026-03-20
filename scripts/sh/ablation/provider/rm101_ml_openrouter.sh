#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../_common.sh"

phmga_enter_repo "$SCRIPT_DIR"
# Active OpenRouter Stage B comparison wrapper.
phmga_prepare_fixed_provider "openrouter" "z-ai/glm-4.5-air:free"

OUTPUT_DIR="${PHMGA_OUTPUT_DIR:-artifacts/paper/rm101_ml_openrouter_glm_v1}"

echo "Running RM101 ML OpenRouter backend comparison..."
phmga_log_provider_choice
python main.py +runs=rm101_ml "${PHMGA_PROVIDER_ARGS[@]}" runtime.output_dir="$OUTPUT_DIR"
phmga_assert_file "$OUTPUT_DIR/final_report.md"
echo "Completed OpenRouter backend comparison: $OUTPUT_DIR"
echo "Next: record the backend comparison result in doc/experiments/01_result_ledger.md"
