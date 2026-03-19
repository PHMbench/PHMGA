#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

phmga_enter_repo "$SCRIPT_DIR"
phmga_prepare_optional_provider "codex_cli" "gpt-5.3-codex"

PREREQ="artifacts/paper/ottawa_ml_pilot_v1/final_report.md"
OUTPUT_DIR="${PHMGA_OUTPUT_DIR:-artifacts/paper/ottawa_ml_main_v1}"

phmga_assert_file "$PREREQ"
phmga_log_provider_choice
python main.py +runs=ottawa_ml "${PHMGA_PROVIDER_ARGS[@]}" runtime.output_dir="$OUTPUT_DIR"

phmga_assert_file "$OUTPUT_DIR/final_report.md"
echo "Completed: $OUTPUT_DIR"
echo "Next: record the result in doc/experiments/01_result_ledger.md"
