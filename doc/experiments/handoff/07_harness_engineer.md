# Ticket: harness-engineer

## Goal

复核 Stage B 的 artifact contract 与 feature separability gate，并决定哪些 row 有资格进入 backend selection。

## Allowed Inputs

- `doc/experiments/handoff/results/*.md`
- `artifacts/paper/<experiment_id>/`
- `doc/experiments/01_result_ledger.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`

## Worker ID

- `harness-engineer`

## Scope

只复核以下 4 行：

- `ottawa_ml_codex_v1`
- `ottawa_ml_openrouter_v1`
- `rm101_ml_codex_v1`
- `rm101_ml_openrouter_v1`

## Success Criteria

- 对每个 Stage B row 明确写出：
  - `artifact_contract_pass`
  - `feature_separability_pass`
  - `selection_eligible`
- 只有在 merge checklist 条件满足时，才允许把 row 标为 `selection_eligible=yes`

## Failure Rule

- harness engineer 不跑 wrapper
- 不更新 `doc/experiments/02_main_tables.md`
- 不直接决定最终 winner；只负责 gate 复核
- 不允许改 worker 填写的 accuracy / macro_f1 原始值
