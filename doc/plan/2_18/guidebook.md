# PHMGA 实验执行指导书（v2.18，3 模型收口版）

本手册用于外部智能体协作执行，覆盖：
- `3 LLM` 矩阵执行（`D1/D2 × A0/A1/A2 = 18 combos`）
- 失败分流与重跑
- DAG -> TSPN 续训
- teammate 并行分工与交接
- 可直接放入论文的中期结果口径

---

## 1) 环境与前置

- 默认环境：`conda run -n agent`
- 依赖与阻断检查：`doc/plan/2_18/env_lock.md`
- LLM 来源优先级：
  - `case.yaml` 含 `llm` 块时，`provider/model` 以 `case.yaml` 为唯一真值
  - `.env` 仅承载 `*_API_KEY` 与 `*_BASE/*_BASE_URL`
- 入口：
  - `python main.py preflight --config <yaml>`
  - `python main.py case1 --config <yaml>`

数据配置基线：
- Ottawa：`config/tspn_case_exp_ottawa.yaml`
- RM101：`config/case_exp_gearbox_rm101.yaml`

---

## 2) 三个实验定义（M1~M3）

- `M1`: `gemini-2.5-flash`（`run_m1_gemini25_full_matrix.sh`）
- `M2`: `gemini-3-flash-preview`（`run_m2_gemini3_full_matrix.sh`）
- `M3`: `GLM-4.7-Flash`（`run_m3_glm47_full_matrix.sh`）

说明：
- `GLM-4.5` 已下线，不在 active matrix。

---

## 3) 调度脚本体系（sh 主导）

### 3.1 最小执行单元
- `scripts/paper/run_combo.sh`
- 流程：resolve config（写入 `llm`）-> preflight -> case1 -> discover artifacts -> append manifest
- 训练档位：`--train-profile fast|standard|highacc`

### 3.2 单 LLM 全矩阵
- `scripts/paper/run_llm_full_matrix.sh`
- 固定循环：
  - datasets: `ottawa`, `rm101`
  - ablations: `A0_full`, `A1_no_reflect`, `A2_no_prior`

### 3.3 失败重跑
- `scripts/paper/rerun_failed_from_manifest.sh --manifest <manifest_dedup.jsonl>`
- 口径：同一 `combo` 只认最新一条记录。

---

## 4) 三模型简化可跑检查（Can-Run Gate）

### Gate-A：LLM 联通
- 先走 `zai SDK`（GLM-4.7）或 provider 直连 smoke。
- 再走 PHMGA 路由 `get_llm().invoke("OK")`。
- 通过标准：返回非空、无 `4xx/5xx`。

### Gate-B：单组合 pilot
优先跑 `Ottawa + A0_full`：
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
通过标准：`manifest` 新增且 `status=ok`。

### Gate-C：放行全矩阵
仅 Gate-A/B 通过的模型进入 `run_m*_full_matrix.sh` 全量执行。

---

## 5) 失败分流（执行层）

1. `403 model access`
   - 先核对：模型权限；
   - 再核对：`case llm` 与 `.env` key/base 平台是否一致。
2. `rc=137`
   - 先降载验证链路：`A0 + no_reflect + fast profile`；
   - 链路通过后再回 `standard/highacc`。
3. `preflight failed`
   - 先修路径/依赖/provider，再进 `case1`。
4. `case failed` 但 preflight 正常
   - 优先查 `_logs/*/run.log` 与 `events.jsonl` 首个 ERROR 节点。

---

## 6) DAG -> TSPN 后续训练

### 6.1 built_state 续训
```bash
scripts/paper/run_train_from_built_state.sh \
  --state-pkl <path_to_built_state.pkl> \
  --case-config config/case_exp_gearbox_rm101.yaml \
  --preflight
```

### 6.2 完整 DAGState JSON 续训
```bash
scripts/paper/run_train_from_dag_json.sh \
  --dag-json <path_to_full_dagstate.json> \
  --case-config config/case_exp_gearbox_rm101.yaml \
  --preflight
```

限制：
- `dag-json` 必须是完整 `DAGState`（含 `channels/nodes/leaves`），压缩 `export_json` 不可用。

---

## 7) 结果汇总与证据

```bash
conda run -n agent python scripts/paper/collect_matrix_results.py \
  --manifest save/paper_matrix/<m*>/manifest_dedup.jsonl \
  --output-dir save/paper_matrix/<m*>

conda run -n agent python scripts/paper/generate_analysis_draft.py \
  --manifest save/paper_matrix/<m*>/manifest_dedup.jsonl \
  --output-dir save/paper_matrix/<m*>
```

每位 teammate 必交付：
- `manifest.jsonl`
- `manifest_dedup.jsonl`
- `_logs/*/preflight.log`
- `_logs/*/run.log`
- `paper_main_results.csv`
- `analysis_draft.md`

---

## 8) 当前可复现子集（Interim）

当前中期可用结果主要来自 `M2`：
- Ottawa A0: `val_acc=0.8978`, `test_acc=0.9069`, `val_macro_f1=0.8990`, `test_macro_f1=0.9062`
- Ottawa A1: 与 A0 当前相同
- RM101 A0: `val_acc=0.5417`, `test_acc=0.6198`, `val_macro_f1=0.4340`, `test_macro_f1=0.4584`
- RM101 A1: `val_acc=0.4479`, `test_acc=0.5208`, `val_macro_f1=0.3325`, `test_macro_f1=0.3662`
- RM101 A2: `val_acc=0.5781`, `test_acc=0.6042`, `val_macro_f1=0.4682`, `test_macro_f1=0.4478`

必须声明限制：
- 目前仅 M2 接近完整；
- M1/M3 覆盖不足；
- 不做跨模型最终排序结论。

---

## 9) Teammates 并行执行计划（按 LLM）

- Teammate-1：`run_m1_gemini25_full_matrix.sh`
- Teammate-2：`run_m2_gemini3_full_matrix.sh`
- Teammate-3：`run_m3_glm47_full_matrix.sh`

交接协议（必须）：
1. `error 分类`（`403/137/preflight/runtime`）
2. `下一步重跑参数`（profile/ablation/dataset）
3. 证据路径（preflight/run log）

唯一分发面板：`doc/plan/2_18/teammates_execution_board.md`。

---

## 10) 参考配置与契约

- `config/case_exp_gearbox_rm101.yaml`
- `config/tspn_case_exp_ottawa.yaml`
- `llm` 字段：
  - `llm.provider`
  - `llm.query_generator_model`
  - `llm.phm_model`
  - `llm.reflection_model`
  - `llm.answer_model`
