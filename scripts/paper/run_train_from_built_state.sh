#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/run_train_from_built_state.sh \
    --state-pkl <path> --case-config <yaml> \
    [--output-root <dir>] [--run-name <name>] [--preflight] [--allow-unverified-state] [--dry-run]
USAGE
}

state_pkl=""
case_config=""
output_root=""
run_name=""
preflight=0
allow_unverified=0
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --state-pkl) state_pkl="$2"; shift 2 ;;
    --case-config) case_config="$2"; shift 2 ;;
    --output-root) output_root="$2"; shift 2 ;;
    --run-name) run_name="$2"; shift 2 ;;
    --preflight) preflight=1; shift ;;
    --allow-unverified-state) allow_unverified=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${state_pkl}" && -n "${case_config}" ]] || { echo "Missing required args." >&2; usage >&2; exit 2; }

cmd=(
  conda run -n agent python "scripts/paper/train_tspn_from_saved_dag.py"
  --state-pkl "${state_pkl}"
  --case-config "${case_config}"
)
[[ -n "${output_root}" ]] && cmd+=(--output-root "${output_root}")
[[ -n "${run_name}" ]] && cmd+=(--run-name "${run_name}")
[[ "${preflight}" -eq 1 ]] && cmd+=(--preflight)
[[ "${allow_unverified}" -eq 1 ]] && cmd+=(--allow-unverified-state)
[[ "${dry_run}" -eq 1 ]] && cmd+=(--dry-run)

echo "[run] ${cmd[*]}"
"${cmd[@]}"
