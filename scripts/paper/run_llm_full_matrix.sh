#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/run_llm_full_matrix.sh \
    --llm-tag <tag> --provider <provider> --model <model> \
    [--output-root save/paper_matrix/<tag>] [--train-profile fast|standard|highacc] [--env agent] [--dry-run]
USAGE
}

llm_tag=""
provider=""
model=""
output_root=""
train_profile=""
conda_env="agent"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-tag) llm_tag="$2"; shift 2 ;;
    --provider) provider="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --train-profile) train_profile="$2"; shift 2 ;;
    --env) conda_env="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${llm_tag}" && -n "${provider}" && -n "${model}" ]] || {
  echo "Missing required arguments." >&2
  usage >&2
  exit 2
}

if [[ -z "${output_root}" ]]; then
  output_root="save/paper_matrix/${llm_tag}"
fi

datasets=(
  "ottawa:config/tspn_case_exp_ottawa.yaml"
  "rm101:config/case_exp_gearbox_rm101.yaml"
)
ablations=(
  "A0_full:full"
  "A1_no_reflect:no_reflect"
  "A2_no_prior:no_prior"
)

failures=0
for ds in "${datasets[@]}"; do
  dataset_tag="${ds%%:*}"
  case_cfg="${ds#*:}"
  for ab in "${ablations[@]}"; do
    ablation_tag="${ab%%:*}"
    ablation_mode="${ab#*:}"
    cmd=(
      "${SCRIPT_DIR}/run_combo.sh"
      --llm-tag "${llm_tag}"
      --provider "${provider}"
      --model "${model}"
      --dataset-tag "${dataset_tag}"
      --case-config "${case_cfg}"
      --ablation-tag "${ablation_tag}"
      --ablation-mode "${ablation_mode}"
      --output-root "${output_root}"
      --env "${conda_env}"
    )
    if [[ -n "${train_profile}" ]]; then
      cmd+=(--train-profile "${train_profile}")
    fi
    if [[ "${dry_run}" -eq 1 ]]; then
      cmd+=(--dry-run)
    fi
    echo "[matrix] ${cmd[*]}"
    set +e
    "${cmd[@]}"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      failures=$((failures + 1))
      echo "[matrix][warn] combo failed with rc=${rc}: ${dataset_tag}/${ablation_tag}" >&2
    fi
  done
done

if [[ "${failures}" -gt 0 ]]; then
  echo "[matrix] completed with ${failures} failed combos." >&2
  exit 1
fi
