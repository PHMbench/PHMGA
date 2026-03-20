# Execution Protocol

本文件是把 PHMGA 实验交给其他 Codex CLI worker 时的唯一执行制度说明。

## Authority

worker、harness engineer 和 coordinator 只依赖以下材料：

- `doc/experiments/00_manual_runbook.md`
- `doc/experiments/01_result_ledger.md`
- `scripts/sh/README.md`
- 本文件
- `README.md`

冲突时优先级固定为：

1. `doc/experiments/00_manual_runbook.md`
2. `doc/experiments/01_result_ledger.md`
3. `scripts/sh/README.md`
4. 本文件
5. `README.md`

## Worker Tool Vs Experiment Backend

- `worker tool = Codex CLI`
- `experiment backend = active/selected backend tuple`

这两者不能混写。

## Fixed Experiment Facts

- `ml` 是 canonical diagnosis mainline
- `torch` 只做 path comparison
- Stage B 只比较 active backend set
- Stage C / D 只围绕 `selected_global_best_backend`
- 当前 shell wrapper 默认 tuple 仍是 `codex_cli / gpt-5.3-codex`
- 如果 selected backend 与 wrapper 默认值不同，worker 必须通过：
  - `PHMGA_LLM_PROVIDER`
  - `PHMGA_LLM_MODEL`
  显式覆盖

## Roles

### Run workers

- `pilot-owner`
- `main-owner-ottawa`
- `main-owner-rm101`
- `ablation-owner-ml`
- `ablation-owner-torch`
- `backend-comparison-owner`

### Harness engineer

- 只负责 gate 复核，不跑 wrapper，不选 backend winner
- 重点复核：
  - `artifact_contract_pass`
  - `feature_separability_pass`
  - `selection_eligible`

### Coordinator

- 只负责：
  - 合并 worker 报告与 ledger
  - 在通过 gate 的 Stage B row 中选择 `selected_global_best_backend`
  - 更新 `doc/experiments/02_main_tables.md`

## Environment And Writeback Order

- 所有 worker 在同一台机器、同一个仓库工作区、同一个 `.venv` 中运行
- Formal Main 与 method ablation 需要：
  - `codex` CLI 已安装
  - `codex login` 已完成
- OpenRouter comparison 额外需要：
  - `OPENROUTER_API_KEY`
- 输出目录固定为：
  - `artifacts/paper/<experiment_id>/`

固定写回顺序：

1. 运行 wrapper 或 runbook 命令
2. 检查目标 artifacts
3. 先写 `doc/experiments/handoff/results/<experiment_id>.md`
4. 再更新 `doc/experiments/01_result_ledger.md`
5. harness engineer 复核 gate
6. coordinator 最后更新 `selected_global_best_backend` 与主表

## Gates

字段解释基线固定参考：

- `doc/experiments/examples/feature_separability_summary.example.json`

### Artifact Contract Gate

以下文件缺任一项，则：

- `artifact_contract_pass=fail`
- `keep` 不得为 `accept`

硬门槛 artifacts：

- `validated_dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `feature_list.json`
- `feature_separability_summary.json`
- `artifact_index.json`
- `metrics.json`
- `final_report.md`

### Required Evidence Gate

- `progress_record` 缺失时，最多记 `needs_rerun`
- worker 报告与 ledger 不一致时，不得直接进入主表

### Feature Separability Gate

仅对 `ml` comparison / main / ml ablation 执行。最小要求：

- feature pipeline 非空
- 不得出现明显全零、常数或塌缩特征主导
- 必须有一个明确的 separability 结论

如果这些条件不满足：

- `feature_separability_pass=fail`
- Stage B row 不得记为 `selection_eligible=yes`

## Acceptance Rules

### accept

- 硬门槛 artifact 完整
- required evidence 完整
- worker 报告与 ledger 一致
- 没有 provider/model 越权行为

### reject

- 运行失败
- artifact 不完整
- separability 证据不成立
- worker 越权改了不允许的配置
- provider/model 与 ticket 不一致

### needs_rerun

- artifact 存在但不完整
- worker 报告与 ledger 不一致
- 缺 `progress_record`
- 结果文件存在明显缺项，但尚不足以直接判 reject

## Backend Selection Rule

只有同时满足以下条件的 Stage B row，才有资格参与比较：

- `keep=accept`
- `artifact_contract_pass=pass`
- `feature_separability_pass=pass`
- `selection_eligible=yes`

选择顺序固定为：

1. 比较 Ottawa + RM101 两条 `ml` comparison row 的 `macro_f1` 均值
2. tie-break 1：`accuracy` 均值
3. tie-break 2：更少的 incident / failed runs
4. tie-break 3：优先当前 active Codex tuple

## Non-Negotiable Rules

- `worker tool != experiment backend`
- worker 不得改 provider 默认、experiment matrix 或 main table 语义
- worker 不得直接更新 `doc/experiments/02_main_tables.md`
- `openrouter/free` 不能进入最终 backend 选择
- `stepfun/step-3.5-flash:free` 的历史失败 row 不参与当前 selection round
- 如果 ledger 冲突，先保留 worker 报告，再由 harness engineer / coordinator 重写 ledger

## Tickets

- `doc/experiments/handoff/01_pilot_owner.md`
- `doc/experiments/handoff/02_main_owner_ottawa.md`
- `doc/experiments/handoff/03_main_owner_rm101.md`
- `doc/experiments/handoff/04_ablation_owner_ml.md`
- `doc/experiments/handoff/05_ablation_owner_torch.md`
- `doc/experiments/handoff/06_backend_comparison_owner.md`
- `doc/experiments/handoff/07_harness_engineer.md`
