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

OUTPUT_DIR="artifacts/paper/rm101_torch_gated_v1"

python main.py +runs=rm101_torch model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir="$OUTPUT_DIR"
test -f "$OUTPUT_DIR/final_report.md"
echo "Completed: $OUTPUT_DIR"
echo "Next: record the result in doc/experiments/01_result_ledger.md"
