#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

conda_env="agent"
dry_run=0
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      conda_env="$2"
      args+=("$1" "$2")
      shift 2
      ;;
    --dry-run)
      dry_run=1
      args+=("$1")
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ "${dry_run}" -ne 1 ]]; then
  gate_cfg_rel="save/paper_matrix/m3_openrouter_gpt4o/_resolved_cases/gate_a_rm101_m3.yaml"
  gate_cfg_abs="${REPO_ROOT}/${gate_cfg_rel}"

  conda run --no-capture-output -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" \
    python scripts/paper/check_llm_gate.py \
      --gate a1 \
      --provider openrouter \
      --model openai/gpt-4o

  conda run --no-capture-output -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" \
    python scripts/paper/resolve_case_config.py \
      --base-config config/case_exp_gearbox_rm101.yaml \
      --out-config "${gate_cfg_abs}" \
      --case-name gate_a_rm101_m3 \
      --save-root save/paper_matrix/m3_openrouter_gpt4o \
      --provider openrouter \
      --model openai/gpt-4o \
      --ablation-mode full \
      --train-backend tspn

  conda run --no-capture-output -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" \
    python scripts/paper/check_llm_gate.py \
      --gate a2 \
      --config "${gate_cfg_abs}"
fi

exec "${SCRIPT_DIR}/run_llm_full_matrix.sh" \
  --llm-tag "m3_openrouter_gpt4o" \
  --provider "openrouter" \
  --model "openai/gpt-4o" \
  --output-root "save/paper_matrix/m3_openrouter_gpt4o" \
  "${args[@]}"
