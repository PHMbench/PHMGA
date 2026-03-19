# Ticket: pilot-owner

## Goal

运行全部 4 个 pilot，确认 smoke 链路和 artifact 完整性。

## Allowed Commands

```bash
./scripts/sh/pilot/01_ottawa_ml_pilot.sh
./scripts/sh/pilot/02_ottawa_torch_pilot.sh
./scripts/sh/pilot/03_rm101_ml_pilot.sh
./scripts/sh/pilot/04_rm101_torch_pilot.sh
```

## Expected Experiment IDs

- `ottawa_ml_pilot_v1`
- `ottawa_torch_pilot_v1`
- `rm101_ml_pilot_v1`
- `rm101_torch_pilot_v1`

## Worker ID

- `pilot-owner`

## Result Files

- `doc/experiments/handoff/results/ottawa_ml_pilot_v1.md`
- `doc/experiments/handoff/results/ottawa_torch_pilot_v1.md`
- `doc/experiments/handoff/results/rm101_ml_pilot_v1.md`
- `doc/experiments/handoff/results/rm101_torch_pilot_v1.md`

## Success Criteria

- 每个输出目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md` 中对应 row

## Failure Rule

- 失败时登记 `keep=reject`
- `note` 写明失败阶段与错误摘要
- 不改配置、不补调参
- 只允许修改本 ticket 对应的 4 行 ledger
