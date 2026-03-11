#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_llm_full_matrix.sh" \
  --llm-tag "m2_openrouter_claude35" \
  --provider "openrouter" \
  --model "anthropic/claude-3.5-sonnet" \
  --output-root "save/paper_matrix/m2_openrouter_claude35" \
  "$@"
