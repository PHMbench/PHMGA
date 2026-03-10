# PHMGA 2.18 执行入口（3 模型 active 版）

本目录用于外部 Agent 直接执行 2.18 矩阵实验与中期论文整理。

## 当前口径

- Active matrix：`3 LLM × 2 Dataset × 3 Ablation = 18 combos`
- Active LLM：
  - `M1 = gemini-2.5-flash`
  - `M2 = gemini-3-flash-preview`
  - `M3 = GLM-4.7-Flash`
- `GLM-4.5`：已下线（仅保留历史结果归档，不参与当前执行）

## 当前执行状态总览（latest manifest snapshot）

| Model                         | Script                             | Success Coverage | Status       | Main Blocking |
| ----------------------------- | ---------------------------------- | ---------------- | ------------ | ------------- |
| M1 `gemini-2.5-flash`       | `run_m1_gemini25_full_matrix.sh` | `4/6`          | Partial      | `S3: rm101 两个组合未收敛` |
| M2 `gemini-3-flash-preview` | `run_m2_gemini3_full_matrix.sh`  | `6/6`          | Complete     | 无 |
| M3 `GLM-4.7-Flash`          | `run_m3_glm47_full_matrix.sh`    | `0/0`          | Pending Gate | 未生成 active manifest |

> 口径：按 `manifest.jsonl` 最新去重组合（同一 combo 取最后一次记录）。

## 文档导航

### Runbook（怎么跑）

- `guidebook.md`：总执行手册（矩阵、续训、分工、交接）
- `real_matrix_run_plan.md`：外部 Agent 一步步执行流程（含 Can-Run Gate）
- `env_lock.md`：环境与 GLM-4.7 双重联通检查（`zai` + PHMGA）

### Interim Results（现在能写什么）

- `analysis_methods.md`：论文中期结果写作方法、模板、限制口径

### Recovery（跑不通如何修）

- `perf_fix_plan_rm101.md`：RM101 性能恢复与分层诊断
- `teammates_execution_board.md`：3 人并行任务看板（唯一分发面板）

## 协作入口（建议顺序）

1. 读 `env_lock.md` 完成 3 模型 Gate-A（联通）。
2. 读 `real_matrix_run_plan.md` 执行 Gate-B（单组合 pilot）。
3. 通过后按 `teammates_execution_board.md` 分工并行。
4. 产出后按 `analysis_methods.md` 生成论文中期文本。

## 关联脚本（`scripts/paper/`）

- `run_m1_gemini25_full_matrix.sh`
- `run_m2_gemini3_full_matrix.sh`
- `run_m3_glm47_full_matrix.sh`
- `run_combo.sh`
- `run_train_from_built_state.sh`
- `run_train_from_dag_json.sh`
- `rerun_failed_from_manifest.sh`
- `collect_matrix_results.py`
- `generate_analysis_draft.py`
