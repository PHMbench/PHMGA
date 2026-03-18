#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
while [ "$ROOT_DIR" != "/" ] && { [ ! -f "$ROOT_DIR/main.py" ] || [ ! -f "$ROOT_DIR/README.md" ]; }; do
  ROOT_DIR="$(cd "$ROOT_DIR/.." && pwd)"
done

if [ ! -f "$ROOT_DIR/main.py" ]; then
  echo "Repository root not found."
  exit 1
fi

cd "$ROOT_DIR"

OUTPUT_DIR="artifacts/paper/ottawa_ml_openrouter_v1"

python main.py +runs=ottawa_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir="$OUTPUT_DIR"
test -f "$OUTPUT_DIR/final_report.md"
echo "Completed: $OUTPUT_DIR"
echo "Next: record the result in doc/experiments/01_result_ledger.md"
