# Ticket: backend-comparison-owner

## Goal

运行 Stage B backend comparison，验证 PHMGA 全链路、artifact contract 和 feature separability gate。

## Allowed Commands

```bash
./scripts/sh/ablation/provider/ottawa_ml_codex.sh
./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh
./scripts/sh/ablation/provider/rm101_ml_codex.sh
./scripts/sh/ablation/provider/rm101_ml_openrouter.sh
```

## Expected Experiment IDs

- `ottawa_ml_codex_v1`
- `ottawa_ml_openrouter_v1`
- `rm101_ml_codex_v1`
- `rm101_ml_openrouter_v1`

## Worker ID

- `backend-comparison-owner`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_codex_v1.md`
- `doc/experiments/handoff/results/ottawa_ml_openrouter_v1.md`
- `doc/experiments/handoff/results/rm101_ml_codex_v1.md`
- `doc/experiments/handoff/results/rm101_ml_openrouter_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- `OPENROUTER_API_KEY` 已设置
- 对应 dataset 的 pilot 已完成并记账
- 当前 active Stage B set 以 `doc/experiments/01_result_ledger.md` 顶部 YAML block 为准

## Success Criteria

- 所有成功运行的目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md`
- ledger 中：
  - 4 行都保持 `backend comparison candidate`
  - 不把任何 comparison row 写进 main rows
  - 普通 worker 不直接把 `selection_eligible` 写成最终事实

## Failure Rule

- 不改 provider family
- 不把 OpenRouter 结果写进 main rows
- 不把 Codex comparison row 写成 frozen main
- 失败时记 `reject`，并写明是 `artifact gate`、`separability gate` 还是 runtime 失败
- 只允许修改本 ticket 对应的 4 行 comparison ledger
