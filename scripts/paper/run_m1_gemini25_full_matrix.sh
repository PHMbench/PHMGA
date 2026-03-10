#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_llm_full_matrix.sh" \
  --llm-tag "m1_gemini25" \
  --provider "openai_compatible" \
  --model "gemini-2.5-flash" \
  --output-root "save/paper_matrix/m1_gemini25" \
  "$@"
