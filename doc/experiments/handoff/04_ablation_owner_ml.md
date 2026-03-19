# Ticket: ablation-owner-ml

## Goal

运行 ML 路径的 method ablation。

## Allowed Commands

```bash
./scripts/sh/ablation/output_policy/ottawa_ml_intermediate.sh
./scripts/sh/ablation/output_policy/rm101_ml_intermediate.sh
```

## Expected Experiment IDs

- `ottawa_ml_intermediate_v1`
- `rm101_ml_intermediate_v1`

## Worker ID

- `ablation-owner-ml`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_intermediate_v1.md`
- `doc/experiments/handoff/results/rm101_ml_intermediate_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- 对应 dataset 的 Formal Main 已完成并记账

## Success Criteria

- 两个输出目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md` 中对应 ablation row

## Failure Rule

- 不改 `output_policy` 之外的配置
- 不覆盖 provider/model
- 失败时记 `reject`
- 只允许修改本 ticket 对应的 2 行 ablation ledger
- 不得把这两行改写成 main result
