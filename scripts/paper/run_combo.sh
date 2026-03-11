#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/run_combo.sh \
    --llm-tag <tag> --provider <provider> --model <model> \
    --dataset-tag <tag> --case-config <yaml> \
    --ablation-tag <tag> --ablation-mode <mode> \
    --output-root <dir> [--train-profile fast|standard|highacc] [--env agent] [--dry-run]
USAGE
}

llm_tag=""
provider=""
model=""
dataset_tag=""
case_config=""
ablation_tag=""
ablation_mode=""
output_root=""
train_profile=""
conda_env="agent"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-tag) llm_tag="$2"; shift 2 ;;
    --provider) provider="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --dataset-tag) dataset_tag="$2"; shift 2 ;;
    --case-config) case_config="$2"; shift 2 ;;
    --ablation-tag) ablation_tag="$2"; shift 2 ;;
    --ablation-mode) ablation_mode="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --train-profile) train_profile="$2"; shift 2 ;;
    --env) conda_env="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${llm_tag}" && -n "${provider}" && -n "${model}" && -n "${dataset_tag}" && -n "${case_config}" && -n "${ablation_tag}" && -n "${ablation_mode}" && -n "${output_root}" ]] || {
  echo "Missing required arguments." >&2
  usage >&2
  exit 2
}

cd "${REPO_ROOT}"

case_config_abs="${case_config}"
if [[ "${case_config_abs}" != /* ]]; then
  case_config_abs="${REPO_ROOT}/${case_config}"
fi
[[ -f "${case_config_abs}" ]] || { echo "Case config not found: ${case_config_abs}" >&2; exit 2; }

output_root_abs="${output_root}"
if [[ "${output_root_abs}" != /* ]]; then
  output_root_abs="${REPO_ROOT}/${output_root}"
fi
mkdir -p "${output_root_abs}/_resolved_cases" "${output_root_abs}/_logs"

combo="${dataset_tag}__${llm_tag}__${ablation_tag}"
case_name="paper_${combo}"
manifest_path="${output_root_abs}/manifest.jsonl"
resolved_cfg="${output_root_abs}/_resolved_cases/${combo}.yaml"
case_dir="${output_root_abs}/${case_name}"
log_dir="${output_root_abs}/_logs/${combo}"
mkdir -p "${log_dir}"
preflight_log="${log_dir}/preflight.log"
run_log="${log_dir}/run.log"

# Source .env to get API keys for preflight/case execution.
# For explicit connectivity gates, use scripts/paper/check_llm_gate.py (load_dotenv-based).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

base_env=(
  "FAKE_LLM=false"
  "PHM_REPORT_MODE=auto"
  "PHM_ABLATION_MODE=${ablation_mode}"
)
# Add OpenRouter variables if set
if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  base_env+=("OPENROUTER_API_KEY=${OPENROUTER_API_KEY}")
fi
if [[ -n "${OPENROUTER_BASE_URL:-}" ]]; then
  base_env+=("OPENROUTER_BASE_URL=${OPENROUTER_BASE_URL}")
fi

resolve_cmd=(
  conda run -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" python scripts/paper/resolve_case_config.py
  --base-config "${case_config_abs}"
  --out-config "${resolved_cfg}"
  --case-name "${case_name}"
  --save-root "${output_root_abs}"
  --provider "${provider}"
  --model "${model}"
  --ablation-mode "${ablation_mode}"
  --train-backend "tspn"
  --allow-test-labels "true"
)
if [[ -n "${train_profile}" ]]; then
  resolve_cmd+=(--train-profile "${train_profile}")
fi

preflight_cmd=(
  conda run -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" python main.py preflight --config "${resolved_cfg}"
)
run_cmd=(
  conda run --no-capture-output -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" python main.py --config-dir "$(dirname -- "${resolved_cfg}")" --config-name "$(basename -- "${resolved_cfg}" .yaml)" --case case1
)
discover_cmd=(
  conda run -n "${conda_env}" -v PYTHONPATH="${REPO_ROOT}" python scripts/paper/discover_run_artifacts.py --case-dir "${case_dir}"
)

echo "[combo] ${combo}"
echo "[resolve] ${resolve_cmd[*]}"
echo "[preflight] ${preflight_cmd[*]}"
echo "[run] ${run_cmd[*]}"

if [[ "${dry_run}" -eq 1 ]]; then
  exit 0
fi

"${resolve_cmd[@]}"

set +e
env "${base_env[@]}" "${preflight_cmd[@]}" >"${preflight_log}" 2>&1
pre_rc=$?
set -e

status="preflight_failed"
run_rc=0
final_rc=0

if [[ "${pre_rc}" -eq 0 ]]; then
  set +e
  env "${base_env[@]}" "${run_cmd[@]}" >"${run_log}" 2>&1
  run_rc=$?
  set -e
  if [[ "${run_rc}" -eq 0 ]]; then
    status="ok"
    final_rc=0
  else
    status="failed"
    final_rc="${run_rc}"
  fi
else
  final_rc="${pre_rc}"
fi

artifact_json="$(env "${base_env[@]}" "${discover_cmd[@]}" 2>/dev/null || echo '{}')"
run_tag="$(date +%Y%m%d-%H%M%S)"

python - "${manifest_path}" "${artifact_json}" "${run_tag}" "${combo}" "${llm_tag}" "${provider}" "${model}" "${dataset_tag}" "${ablation_tag}" "${ablation_mode}" "${status}" "${final_rc}" "${resolved_cfg}" "${preflight_log}" "${run_log}" "${case_dir}" "${output_root_abs}" "${train_profile}" "${case_config_abs}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
artifacts_raw = sys.argv[2]
artifacts = {}
try:
    artifacts = json.loads(artifacts_raw) if artifacts_raw else {}
except Exception:
    artifacts = {}

record = {
    "run_tag": sys.argv[3],
    "combo": sys.argv[4],
    "llm": sys.argv[5],
    "provider": sys.argv[6],
    "model": sys.argv[7],
    "dataset": sys.argv[8],
    "ablation": sys.argv[9],
    "ablation_mode": sys.argv[10],
    "status": sys.argv[11],
    "return_code": int(sys.argv[12]),
    "resolved_case_config": sys.argv[13],
    "preflight_log": sys.argv[14],
    "run_log": sys.argv[15],
    "case_dir": sys.argv[16],
    "output_root": sys.argv[17],
    "train_profile": sys.argv[18],
    "case_config": sys.argv[19],
}
if isinstance(artifacts, dict):
    record.update(artifacts)
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print(json.dumps(record, ensure_ascii=False))
PY

if [[ "${status}" != "ok" ]]; then
  exit "${final_rc:-1}"
fi
