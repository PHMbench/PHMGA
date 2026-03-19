# Ticket: codex-main-owner-ottawa

## Goal

运行 Ottawa 的 Codex Formal Main。

## Allowed Commands

```bash
./scripts/sh/main/01_ottawa_ml_main.sh
./scripts/sh/main/02_ottawa_torch_main.sh
```

## Expected Experiment IDs

- `ottawa_ml_main_v1`
- `ottawa_torch_main_v1`

## Worker ID

- `codex-main-owner-ottawa`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_main_v1.md`
- `doc/experiments/handoff/results/ottawa_torch_main_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- Ottawa 对应 pilot 已完成并记账

## Success Criteria

- 两个输出目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md` 中对应 main row
- `note` 保持 `formal main frozen default: codex_cli / gpt-5.3-codex`

## Failure Rule

- 不覆盖 `PHMGA_LLM_PROVIDER`
- 不覆盖 `PHMGA_LLM_MODEL`
- 失败时记 `reject`，并写明 provider / stage / exception 摘要
- 只允许修改本 ticket 对应的 2 行 main ledger
- 不得改 `note` 的 formal-main 语义
