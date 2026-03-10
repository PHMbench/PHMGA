# PHMGA v2.18 真实矩阵执行计划（3 模型 / 18 组合）

## Summary
目标：在无历史上下文条件下，完成 `3 LLM × 2 Dataset × 3 Ablation = 18 combos`。  
执行策略：先过 Can-Run Gate，再进入 6 组合矩阵。  
结果口径：先产出 interim（当前可复现子集），后续补齐全矩阵。

---

## 一、执行契约（固定）

- 运行环境：`conda run -n agent`
- 输出目录：
  - `save/paper_matrix/m1_gemini25`
  - `save/paper_matrix/m2_gemini3`
  - `save/paper_matrix/m3_glm47`
- 入口脚本：
  - `scripts/paper/run_m1_gemini25_full_matrix.sh`
  - `scripts/paper/run_m2_gemini3_full_matrix.sh`
  - `scripts/paper/run_m3_glm47_full_matrix.sh`
- `provider/model` 来源：`run_combo.sh -> resolve_case_config.py` 写入 `resolved case.yaml` 的 `llm` 字段；运行阶段不注入 `LLM_PROVIDER/QUERY_GENERATOR_MODEL`。

---

## 二、执行步骤

## Step -1：Can-Run Gate（3 模型简化可跑检查）

### Gate-A：模型联通
每个模型先跑 1 次 LLM smoke（详见 `env_lock.md`）。  
通过标准：无 `4xx/5xx`，返回非空文本。

推荐统一入口（自动 `load_dotenv`）：
```bash
conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a1 --provider glm --model glm-4.7-flash

conda run -n agent -v PYTHONPATH=$(pwd) python scripts/paper/check_llm_gate.py \
  --gate a2 --config save/paper_matrix/m3_glm47/_resolved_cases/gate_a_rm101_m3.yaml
```

### Gate-B：单组合 pilot（推荐 Ottawa + A0）
```bash
scripts/paper/run_combo.sh \
  --llm-tag <m*> \
  --provider <provider> \
  --model <model> \
  --dataset-tag ottawa \
  --case-config config/tspn_case_exp_ottawa.yaml \
  --ablation-tag A0_full \
  --ablation-mode full \
  --output-root save/paper_matrix/<m*> \
  --env agent
```
通过标准：
- `manifest.jsonl` 新增 1 条；
- 该条 `status=ok`；
- `run_dir` 可访问。

### Gate-C：放行 6 组合
仅通过 Gate-A/B 的模型进入 `run_m*_full_matrix.sh`。

---

## Step 0：前置健康检查（阻断）
```bash
bash -n scripts/paper/run_combo.sh scripts/paper/run_llm_full_matrix.sh \
  scripts/paper/run_m1_gemini25_full_matrix.sh scripts/paper/run_m2_gemini3_full_matrix.sh \
  scripts/paper/run_m3_glm47_full_matrix.sh scripts/paper/rerun_failed_from_manifest.sh

scripts/paper/run_m1_gemini25_full_matrix.sh --dry-run
scripts/paper/run_m2_gemini3_full_matrix.sh --dry-run
scripts/paper/run_m3_glm47_full_matrix.sh --dry-run
```
阻断条件：
- 脚本语法失败；
- 干跑非 6 组合；
- Ottawa/RM101 case 配置不可读。

---

## Step 1：三终端并行执行

**T1**
```bash
scripts/paper/run_m1_gemini25_full_matrix.sh
```

**T2**
```bash
scripts/paper/run_m2_gemini3_full_matrix.sh
```

**T3**
```bash
scripts/paper/run_m3_glm47_full_matrix.sh
```

---

## Step 2：目录级验收
```bash
python - <<'PY'
import json, pathlib
roots = [
 "save/paper_matrix/m1_gemini25",
 "save/paper_matrix/m2_gemini3",
 "save/paper_matrix/m3_glm47",
]
for r in roots:
    p = pathlib.Path(r) / "manifest.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []
    ok = sum(1 for x in rows if x.get("status") == "ok")
    print(r, "rows=", len(rows), "ok=", ok, "fail=", len(rows) - ok)
PY
```

---

## Step 3：失败分流决策树

### A. `403 model access`
1. 核对 `provider/model` 是否与账号权限匹配。
2. 核对 `resolved_case.yaml` 的 `llm` 与 `.env` key/base 是否同平台。
3. 修正后只重跑失败 combo（不重跑全矩阵）。

### B. `rc=137 (killed)`
1. 先降载验证链路：`A0 + no_reflect + fast profile`。
2. 链路通后再切回 `standard/highacc`。
3. 单模型最多做 2 轮配置调整，超过则转 Recovery 文档记录阻塞。

### C. `preflight_failed`
1. 先修路径/依赖/provider。
2. `preflight` 通过后再进入 `case1`。

---

## Step 4：去重 + 失败重跑

同一 combo 只保留最后一条：
```bash
python - <<'PY'
import json, pathlib
root = pathlib.Path("save/paper_matrix/m3_glm47")
src = root / "manifest.jsonl"
dst = root / "manifest_dedup.jsonl"
rows = [json.loads(x) for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
latest = {}
for r in rows:
    latest[r.get("combo")] = r
with dst.open("w", encoding="utf-8") as f:
    for k in sorted(latest):
        f.write(json.dumps(latest[k], ensure_ascii=False) + "\n")
print(dst)
PY
```

只重跑失败组合：
```bash
scripts/paper/rerun_failed_from_manifest.sh \
  --manifest save/paper_matrix/m3_glm47/manifest_dedup.jsonl \
  --env agent
```

---

## Step 5：汇总与论文草稿

```bash
conda run -n agent python scripts/paper/collect_matrix_results.py \
  --manifest save/paper_matrix/m3_glm47/manifest_dedup.jsonl \
  --output-dir save/paper_matrix/m3_glm47

conda run -n agent python scripts/paper/generate_analysis_draft.py \
  --manifest save/paper_matrix/m3_glm47/manifest_dedup.jsonl \
  --output-dir save/paper_matrix/m3_glm47
```

全模型合并：
```bash
cat \
  save/paper_matrix/m1_gemini25/manifest_dedup.jsonl \
  save/paper_matrix/m2_gemini3/manifest_dedup.jsonl \
  save/paper_matrix/m3_glm47/manifest_dedup.jsonl \
  > save/paper_matrix/all_llm_manifest.jsonl
```

---

## 三、验收标准

1. 每模型至少完成 Gate-A/B 的证据留存。
2. 成功模型给出 `manifest_dedup.jsonl`。
3. 每模型目录有 `paper_main_results.csv` 与 `analysis_draft.md`。
4. 失败模型必须有 `error 分类 + 下一步重跑参数`。

---

## 四、证据留存（必须）

1. 3 模型 Gate-A 输出（stdout+exit code）。
2. 3 模型 Gate-B 的 `manifest` 新增记录。
3. 每模型的 `manifest_dedup.jsonl`。
4. `_logs/*/preflight.log` 与 `_logs/*/run.log`。
5. 汇总产物：`paper_main_results.csv`、`analysis_draft.md`。
