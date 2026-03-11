#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DOTENV_FILE="${REPO_ROOT}/.env"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_case.sh --config <yaml> [--case case1] [--env agent] [--profile tspn_basic] [--dataset Ottawa] [--no-conda] [--dry-run]

Options:
  --config <yaml>   Path to case YAML (required)
  --case <name>     Case module name for main.py (default: case1)
  --env <name>      Conda env name when conda mode is on (default: agent)
  --profile <name>  Optional model profile override (export PHM_MODEL_PROFILE)
  --dataset <name>  Optional dataset override (export PHM_DATASET_NAME)
  --no-conda        Run with current python directly
  --dry-run         Print final command only, do not execute
  -h, --help        Show this message
USAGE
}

read_dotenv_value() {
  local key="$1"
  [[ -f "${DOTENV_FILE}" ]] || return 0
  local line
  line="$(grep -E "^${key}=" "${DOTENV_FILE}" | tail -n 1 || true)"
  [[ -n "${line}" ]] || return 0
  local value="${line#*=}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "${value}"
}

effective_env_value() {
  local key="$1"
  local current="${!key:-}"
  if [[ -n "${current}" ]]; then
    printf '%s' "${current}"
    return 0
  fi
  read_dotenv_value "${key}"
}

base_host_from_url() {
  local url="$1"
  if [[ -z "${url}" ]]; then
    return 0
  fi
  local host="${url#*://}"
  host="${host%%/*}"
  printf '%s' "${host}"
}

read_case_llm_value() {
  local yaml_path="$1"
  local key="$2"
  python - "${yaml_path}" "${key}" <<'PY' || true
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
key = sys.argv[2]
try:
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
except Exception:
    print("")
    raise SystemExit(0)
llm = cfg.get("llm") or {}
value = llm.get(key, "")
print(value if value is not None else "")
PY
}

case_name="case1"
conda_env="agent"
use_conda=1
dry_run=0
config_path=""
model_profile=""
dataset_name=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "Error: --config requires a value." >&2; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "Error: --case requires a value." >&2; exit 2; }
      case_name="$2"
      shift 2
      ;;
    --env)
      [[ $# -ge 2 ]] || { echo "Error: --env requires a value." >&2; exit 2; }
      conda_env="$2"
      shift 2
      ;;
    --profile)
      [[ $# -ge 2 ]] || { echo "Error: --profile requires a value." >&2; exit 2; }
      model_profile="$2"
      shift 2
      ;;
    --dataset)
      [[ $# -ge 2 ]] || { echo "Error: --dataset requires a value." >&2; exit 2; }
      dataset_name="$2"
      shift 2
      ;;
    --no-conda)
      use_conda=0
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n "${config_path}" ]] || { echo "Error: --config is required." >&2; usage >&2; exit 2; }

cd "${REPO_ROOT}"

[[ -f "main.py" ]] || { echo "Error: main.py not found in repo root: ${REPO_ROOT}" >&2; exit 2; }

if [[ "${config_path}" = /* ]]; then
  config_abs="${config_path}"
else
  config_abs="${REPO_ROOT}/${config_path}"
fi

[[ -f "${config_abs}" ]] || { echo "Error: config file not found: ${config_abs}" >&2; exit 2; }

config_dir="$(cd -- "$(dirname -- "${config_abs}")" && pwd)"
config_name="$(basename -- "${config_abs}")"
config_name="${config_name%.*}"

base_cmd=(python main.py --config-dir "${config_dir}" --config-name "${config_name}" --case "${case_name}")

provider="$(read_case_llm_value "${config_abs}" "provider")"
model_name="$(read_case_llm_value "${config_abs}" "query_generator_model")"
provider_source="case_yaml"
if [[ -z "${provider}" ]]; then
  provider="$(effective_env_value LLM_PROVIDER)"
  provider_source="env"
fi
if [[ -z "${model_name}" ]]; then
  model_name="$(effective_env_value QUERY_GENERATOR_MODEL)"
  if [[ -z "${model_name}" ]]; then
    model_name="$(effective_env_value PHM_MODEL)"
  fi
fi

base_url=""
case "${provider}" in
  openrouter)
    base_url="$(effective_env_value OPENROUTER_BASE_URL)"
    ;;
esac
base_host="$(base_host_from_url "${base_url}")"

if [[ "${use_conda}" -eq 1 ]]; then
  command -v conda >/dev/null 2>&1 || { echo "Error: conda not found in PATH." >&2; exit 2; }
  final_cmd=(conda run --no-capture-output -n "${conda_env}" "${base_cmd[@]}")
else
  final_cmd=("${base_cmd[@]}")
fi

printf 'Command:'
printf ' %q' "${final_cmd[@]}"
printf '\n'
echo "LLM_PROVIDER=${provider:-}"
echo "QUERY_GENERATOR_MODEL=${model_name:-}"
echo "LLM_SOURCE=${provider_source}"
echo "BASE_HOST=${base_host:-}"
echo "PHM_MODEL_PROFILE=${model_profile:-}"
echo "PHM_DATASET_NAME=${dataset_name:-}"

if [[ "${dry_run}" -eq 1 ]]; then
  exit 0
fi

if [[ -n "${model_profile}" ]]; then
  export PHM_MODEL_PROFILE="${model_profile}"
fi
if [[ -n "${dataset_name}" ]]; then
  export PHM_DATASET_NAME="${dataset_name}"
fi

"${final_cmd[@]}"
