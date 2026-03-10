#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  scripts/paper/rerun_failed_from_manifest.sh \
    --manifest <manifest_dedup.jsonl|manifest.jsonl> \
    [--env agent] [--dry-run]
USAGE
}

manifest=""
conda_env="agent"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --env) conda_env="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${manifest}" ]] || { echo "--manifest is required" >&2; usage >&2; exit 2; }

cd "${REPO_ROOT}"
manifest_abs="${manifest}"
if [[ "${manifest_abs}" != /* ]]; then
  manifest_abs="${REPO_ROOT}/${manifest_abs}"
fi
[[ -f "${manifest_abs}" ]] || { echo "Manifest not found: ${manifest_abs}" >&2; exit 2; }

mapfile -t combos < <(
  python - "${manifest_abs}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except Exception:
        continue

latest = {}
for row in rows:
    combo = str(row.get("combo") or "")
    if combo:
        latest[combo] = row

def infer_provider_model(llm_tag: str):
    tag = (llm_tag or "").lower()
    if "m3_glm47" in tag:
        return "glm", "GLM-4.7-Flash"
    if "m2_gemini3" in tag:
        return "openai_compatible", "gemini-3-flash-preview"
    if "m1_gemini25" in tag:
        return "openai_compatible", "gemini-2.5-flash"
    return "", ""

for combo, row in sorted(latest.items()):
    status = str(row.get("status") or "")
    if status == "ok":
        continue
    llm_tag = str(row.get("llm") or "")
    active_llm_tags = {"m1_gemini25", "m2_gemini3", "m3_glm47"}
    if llm_tag not in active_llm_tags:
        sys.stderr.write(f"[rerun][skip] inactive_or_legacy_tag llm={llm_tag} combo={combo}\n")
        continue
    provider = str(row.get("provider") or "")
    model = str(row.get("model") or "")
    if not provider or not model:
        p, m = infer_provider_model(llm_tag)
        provider = provider or p
        model = model or m
    dataset = str(row.get("dataset") or "")
    case_config = str(row.get("case_config") or "")
    if not case_config:
        case_config = "config/case_exp_gearbox_rm101.yaml" if dataset == "rm101" else "config/tspn_case_exp_ottawa.yaml"
    output_root = str(row.get("output_root") or (Path(str(row.get("case_dir") or ".")).parent))
    ablation = str(row.get("ablation") or "")
    ablation_mode = str(row.get("ablation_mode") or "")
    train_profile = str(row.get("train_profile") or "")
    if not (llm_tag and provider and model and dataset and case_config and ablation and ablation_mode and output_root):
        continue
    parts = [llm_tag, provider, model, dataset, case_config, ablation, ablation_mode, output_root, train_profile]
    print("\t".join(parts))
PY
)

if [[ "${#combos[@]}" -eq 0 ]]; then
  echo "No failed combos to rerun."
  exit 0
fi

for row in "${combos[@]}"; do
  IFS=$'\t' read -r llm_tag provider model dataset case_config ablation_tag ablation_mode output_root train_profile <<<"${row}"
  cmd=(
    "${SCRIPT_DIR}/run_combo.sh"
    --llm-tag "${llm_tag}"
    --provider "${provider}"
    --model "${model}"
    --dataset-tag "${dataset}"
    --case-config "${case_config}"
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
  echo "[rerun] ${cmd[*]}"
  "${cmd[@]}"
done
