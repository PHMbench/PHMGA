# Ticket: main-owner-rm101

## Goal

运行 RM101 的 Formal Main，使用 `selected_global_best_backend`。

## Allowed Commands

```bash
./scripts/sh/main/03_rm101_ml_main.sh
./scripts/sh/main/04_rm101_torch_main.sh
```

## Expected Experiment IDs

- `rm101_ml_main_v1`
- `rm101_torch_main_v1`

## Worker ID

- `main-owner-rm101`

## Result Files

- `doc/experiments/handoff/results/rm101_ml_main_v1.md`
- `doc/experiments/handoff/results/rm101_torch_main_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- RM101 对应 pilot 已完成并记账
- `doc/experiments/01_result_ledger.md` 顶部 `selected_global_best_backend.selected_from_stage_b=true`

## Success Criteria

- 两个输出目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md` 中对应 main row
- `provider/model` 固定写 selected backend tuple

## Failure Rule

- 如 selected backend 与 wrapper 默认值不同，只允许按 runbook 要求覆盖 `PHMGA_LLM_PROVIDER` / `PHMGA_LLM_MODEL`
- 失败时记 `reject`，并写明 provider / stage / exception 摘要
- 只允许修改本 ticket 对应的 2 行 main ledger
- 不得改 `note` 的 selected-backend 语义
