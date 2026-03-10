#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_llm_full_matrix.sh" \
  --llm-tag "m2_gemini3" \
  --provider "openai_compatible" \
  --model "gemini-3-flash-preview" \
  --output-root "save/paper_matrix/m2_gemini3" \
  "$@"
