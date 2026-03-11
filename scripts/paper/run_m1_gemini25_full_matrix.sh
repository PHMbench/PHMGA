#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_llm_full_matrix.sh" \
  --llm-tag "m1_openrouter_gpt4omini" \
  --provider "openrouter" \
  --model "openai/gpt-4o-mini" \
  --output-root "save/paper_matrix/m1_openrouter_gpt4omini" \
  "$@"
