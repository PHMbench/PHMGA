# 2.22 执行 Runbook（可直接复制）

> 口径：active matrix 仅 `m1_gemini25` / `m2_gemini3` / `m3_glm47`。
> legacy 历史证据见 `doc/plan/2_22/legacy_glm45_archive.md`。

---

## Phase 0 — 基线冻结（只读）

```bash
python - <<'PY'
import json
from pathlib import Path
roots = [
    Path("save/paper_matrix/m1_gemini25"),
    Path("save/paper_matrix/m2_gemini3"),
    Path("save/paper_matrix/m3_glm47"),
]
for root in roots:
    m = root / "manifest.jsonl"
    print("\\n===", root)
    if not m.exists():
        print("manifest: MISSING")
        continue
    latest = {}
    for line in m.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            combo = row.get("combo")
            if combo:
                latest[combo] = row
    ok = sum(1 for r in latest.values() if r.get("status") == "ok")
    print("latest=", len(latest), "ok=", ok, "fail=", len(latest)-ok)
PY
```

---

## Phase 1 — 联通门禁（GLM-4.7）

### Gate-A1：`zai` 直连

```bash
conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a1 \
  --provider glm \
  --model glm-4.7-flash
```

### Gate-A2：PHMGA 路由

```bash
conda run -n agent python scripts/paper/resolve_case_config.py \
  --base-config config/case_exp_gearbox_rm101.yaml \
  --out-config save/paper_matrix/m3_glm47/_resolved_cases/gate_a_rm101_m3.yaml \
  --case-name gate_a_rm101_m3 \
  --save-root save/paper_matrix/m3_glm47 \
  --provider glm \
  --model GLM-4.7-Flash \
  --ablation-mode full \
  --train-backend tspn

conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a2 \
  --config save/paper_matrix/m3_glm47/_resolved_cases/gate_a_rm101_m3.yaml
```

任一失败：按 `S2` 处理，停止 m3 全矩阵放行。

---

## Phase 2 — 先修可跑失败（M2 -> M1）

### 2.1 M2 单失败组合重跑（先 fast）

```bash
scripts/paper/run_combo.sh \
  --llm-tag m2_gemini3 \
  --provider openai_compatible \
  --model gemini-3-flash-preview \
  --dataset-tag ottawa \
  --case-config config/tspn_case_exp_ottawa.yaml \
  --ablation-tag A2_no_prior \
  --ablation-mode no_prior \
  --output-root save/paper_matrix/m2_gemini3 \
  --train-profile fast \
  --env agent
```

### 2.2 M1 全矩阵补齐

```bash
scripts/paper/run_m1_gemini25_full_matrix.sh --env agent
```

若 M1/M2 仍出现 `rc=137`，执行 S1 第二轮（`standard + no_reflect` 对单 combo）。

---

## Phase 3 — M3（GLM-4.7）激活

### 3.1 Pilot

```bash
scripts/paper/run_combo.sh \
  --llm-tag m3_glm47 \
  --provider glm \
  --model GLM-4.7-Flash \
  --dataset-tag ottawa \
  --case-config config/tspn_case_exp_ottawa.yaml \
  --ablation-tag A0_full \
  --ablation-mode full \
  --output-root save/paper_matrix/m3_glm47 \
  --env agent
```

### 3.2 Full matrix（仅 pilot 成功后）

```bash
scripts/paper/run_m3_glm47_full_matrix.sh --env agent
```

若 `403` 持续：标记 `blocked`，停止盲跑并保留证据。

---

## Phase 4 — 去重与汇总

### 4.1 每个 active 目录生成 dedup

```bash
python - <<'PY'
import json
from pathlib import Path
roots = [
    Path("save/paper_matrix/m1_gemini25"),
    Path("save/paper_matrix/m2_gemini3"),
    Path("save/paper_matrix/m3_glm47"),
]
for root in roots:
    src = root / "manifest.jsonl"
    dst = root / "manifest_dedup.jsonl"
    if not src.exists():
        continue
    latest = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            combo = row.get("combo")
            if combo:
                latest[combo] = row
    with dst.open("w", encoding="utf-8") as f:
        for combo in sorted(latest):
            f.write(json.dumps(latest[combo], ensure_ascii=False) + "\\n")
    print("wrote", dst, "rows=", len(latest))
PY
```

### 4.2 合并 active manifest

```bash
cat \
  save/paper_matrix/m1_gemini25/manifest_dedup.jsonl \
  save/paper_matrix/m2_gemini3/manifest_dedup.jsonl \
  save/paper_matrix/m3_glm47/manifest_dedup.jsonl \
  > save/paper_matrix/all_llm_manifest.jsonl
```

### 4.3 生成结果表与草稿

```bash
conda run -n agent python scripts/paper/collect_matrix_results.py \
  --manifest save/paper_matrix/all_llm_manifest.jsonl \
  --output-dir save/paper_matrix

conda run -n agent python scripts/paper/generate_analysis_draft.py \
  --manifest save/paper_matrix/all_llm_manifest.jsonl \
  --output-dir save/paper_matrix
```

---

## 明确禁止

1. 禁止把 `m3_glm45` 或 `m4_glm47` 目录作为 active matrix 重跑。  
2. 禁止将 legacy 失败记录并入 active 成功率。  
