#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_common.sh"

phmga_enter_repo "$SCRIPT_DIR"

OUTPUT_DIR="${PHMGA_OUTPUT_DIR:-artifacts/paper/rm101_ml_pilot_v1}"

echo "Running RM101 ML pilot wrapper..."
echo "Pilot uses offline_stub by design."
python main.py runtime.action=preflight +runs=rm101_ml_test llm.mode=offline_stub
python main.py +runs=rm101_ml_test llm.mode=offline_stub runtime.output_dir="$OUTPUT_DIR"

phmga_assert_file "$OUTPUT_DIR/final_report.md"
echo "Completed: $OUTPUT_DIR"
echo "Next: record the result in doc/experiments/01_result_ledger.md"
