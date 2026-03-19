# Ticket: qualification-owner

## Goal

运行 provider qualification 和 frozen backend sanity check。

## Allowed Commands

```bash
./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh
./scripts/sh/ablation/provider/rm101_ml_openrouter.sh
./scripts/sh/ablation/provider/ottawa_ml_codex.sh
./scripts/sh/ablation/provider/rm101_ml_codex.sh
```

## Expected Experiment IDs

- `ottawa_ml_openrouter_v1`
- `rm101_ml_openrouter_v1`
- `ottawa_ml_codex_v1`
- `rm101_ml_codex_v1`

## Worker ID

- `qualification-owner`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_openrouter_v1.md`
- `doc/experiments/handoff/results/rm101_ml_openrouter_v1.md`
- `doc/experiments/handoff/results/ottawa_ml_codex_v1.md`
- `doc/experiments/handoff/results/rm101_ml_codex_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- `OPENROUTER_API_KEY` 已设置
- 对应 dataset 的 pilot 已完成并记账

## Success Criteria

- 所有成功运行的目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md`
- ledger 中：
  - OpenRouter row 仍保持 `provider qualification candidate`
  - Codex row 仍保持 `formal-main backend sanity check`

## Failure Rule

- 不把 OpenRouter 结果写进 main rows
- 不把 Codex sanity check 写成新的 main result
- 不改 provider family
- 失败时记 `reject`，并写明是 `qualification` 还是 `sanity check`
- 只允许修改本 ticket 对应的 4 行 qualification ledger
- 不得改 OpenRouter / Codex 两类 note 的语义边界
