#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../../_common.sh"

phmga_enter_repo "$SCRIPT_DIR"
phmga_prepare_optional_provider "codex_cli" "gpt-5.3-codex"

OUTPUT_DIR="${PHMGA_OUTPUT_DIR:-artifacts/paper/rm101_ml_intermediate_v1}"

phmga_log_provider_choice
python main.py +runs=rm101_ml "${PHMGA_PROVIDER_ARGS[@]}" model.ml.output_policy=include_intermediate_features runtime.output_dir="$OUTPUT_DIR"
phmga_assert_file "$OUTPUT_DIR/final_report.md"
echo "Completed: $OUTPUT_DIR"
echo "Next: record the result in doc/experiments/01_result_ledger.md"
