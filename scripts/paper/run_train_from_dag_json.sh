#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/run_train_from_dag_json.sh \
    --dag-json <path> --case-config <yaml> \
    [--output-root <dir>] [--run-name <name>] [--preflight] [--dry-run]
USAGE
}

dag_json=""
case_config=""
output_root=""
run_name=""
preflight=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dag-json) dag_json="$2"; shift 2 ;;
    --case-config) case_config="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --run-name) run_name="$2"; shift 2 ;;
    --preflight) preflight=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${dag_json}" && -n "${case_config}" ]] || { echo "Missing required args." >&2; usage >&2; exit 2; }

cmd=(
  conda run -n agent python "scripts/paper/train_tspn_from_saved_dag.py"
  --dag-json "${dag_json}"
  --case-config "${case_config}"
)
[[ -n "${output_root}" ]] && cmd+=(--output-root "${output_root}")
[[ -n "${run_name}" ]] && cmd+=(--run-name "${run_name}")
[[ "${preflight}" -eq 1 ]] && cmd+=(--preflight)
[[ "${dry_run}" -eq 1 ]] && cmd+=(--dry-run)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
