# Codex CLI Worker Handoff

本文件定义把 PHMGA 实验交给其他 Codex CLI worker 时的唯一交接合同。

## Authority

worker 只依赖以下材料：

- `README.md`
- `doc/experiments/00_manual_runbook.md`
- `doc/experiments/01_result_ledger.md`
- `scripts/sh/README.md`
- 本文件

如果这些材料之间出现冲突，优先级固定为：

1. `doc/experiments/00_manual_runbook.md`
2. `doc/experiments/01_result_ledger.md`
3. `scripts/sh/README.md`
4. 本文件
5. `README.md`

## Worker Tool Vs Experiment Backend

- `worker tool = Codex CLI`
  - 其他 agent 统一视为 `Codex CLI worker`
  - 他们用 Codex CLI 接任务、执行 wrapper、写结果报告和 ledger
- `experiment backend = active/selected backend tuple`
  - Stage B 使用 `active_stage_b_set`
  - Stage C / D 使用 `selected_global_best_backend`

这两者不能混写。  
Codex CLI 是外层执行控制面，不等于实验内部一定使用 Codex backend。

## Fixed Experiment Facts

- `ml` 是 canonical diagnosis mainline
- `torch` 只做 path comparison
- Stage B 只比较 active backend set，不做大矩阵
- Stage C / D 只围绕 `selected_global_best_backend`
- `OpenRouter / stepfun/step-3.5-flash:free` 当前是 active OpenRouter comparison candidate
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

- `harness engineer` 不负责跑实验
- 只负责：
  - 检查 artifact contract
  - 检查 feature separability gate
  - 复核 Stage B row 是否有资格记为 `selection_eligible=yes`

### Coordinator

- 只负责：
  - 合并 worker 报告
  - 在通过 gate 的 Stage B row 中选择 `selected_global_best_backend`
  - 更新 `doc/experiments/02_main_tables.md`

## Environment

- 所有 worker 在同一台机器、同一个仓库工作区、同一个 `.venv` 中运行
- Formal Main 与 method ablation 需要：
  - `codex` CLI 已安装
  - `codex login` 已完成
- OpenRouter backend comparison 额外需要：
  - `OPENROUTER_API_KEY`
- 输出目录固定为：
  - `artifacts/paper/<experiment_id>/`

推荐启动前检查：

```bash
cd /home/user/LQ/B_Signal/PHMGA
./scripts/sh/SETUP.sh
which codex
```

backend-comparison worker 额外检查：

```bash
test -n "${OPENROUTER_API_KEY:-}"
```

## Writeback Order

worker 必须先写 worker 结果报告，再更新 ledger。

- worker 必须：
  1. 运行 wrapper 或 runbook 命令
  2. 检查目标 artifact
  3. 先写 `doc/experiments/handoff/results/<experiment_id>.md`
  4. 再更新 `doc/experiments/01_result_ledger.md`
- harness engineer 随后复核：
  - `artifact_contract_pass`
  - `feature_separability_pass`
  - `selection_eligible`
  - 重点检查 `validated_dag.json`、`feature_list.json`、`feature_separability_summary.json`、`artifact_index.json`
- coordinator 最后才允许更新：
  - `selected_global_best_backend`
  - `doc/experiments/02_main_tables.md`

## Failure Rules

- 运行失败后：
  - 不删 artifact
  - 不换 provider/model
  - 先写 `results/<experiment_id>.md`
  - 再在 ledger 中登记 `keep=reject`
- worker 不得：
  - 不得改 provider 默认
  - 改 `config/runs/*.yaml`
  - 改 experiment matrix
  - 改 main table 语义
  - 自行选择 backend winner
- 如果 ledger 写回发生冲突：
  - 先保留 worker 报告
  - 再由 harness engineer / coordinator 重新合并 ledger

## Tickets

- `doc/experiments/handoff/01_pilot_owner.md`
- `doc/experiments/handoff/02_main_owner_ottawa.md`
- `doc/experiments/handoff/03_main_owner_rm101.md`
- `doc/experiments/handoff/04_ablation_owner_ml.md`
- `doc/experiments/handoff/05_ablation_owner_torch.md`
- `doc/experiments/handoff/06_backend_comparison_owner.md`
- `doc/experiments/handoff/07_harness_engineer.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`
