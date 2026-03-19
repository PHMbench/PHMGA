# Ticket: ablation-owner-torch

## Goal

运行 torch 路径的 method ablation。

## Allowed Commands

```bash
./scripts/sh/ablation/runtime/ottawa_torch_module_runtime.sh
./scripts/sh/ablation/runtime/rm101_torch_module_runtime.sh
./scripts/sh/ablation/control/gated/ottawa_torch_gated.sh
./scripts/sh/ablation/control/gated/rm101_torch_gated.sh
./scripts/sh/ablation/control/attention/ottawa_torch_attention.sh
./scripts/sh/ablation/control/attention/rm101_torch_attention.sh
```

## Expected Experiment IDs

- `ottawa_torch_module_runtime_v1`
- `rm101_torch_module_runtime_v1`
- `ottawa_torch_gated_v1`
- `rm101_torch_gated_v1`
- `ottawa_torch_attention_v1`
- `rm101_torch_attention_v1`

## Worker ID

- `ablation-owner-torch`

## Result Files

- `doc/experiments/handoff/results/ottawa_torch_module_runtime_v1.md`
- `doc/experiments/handoff/results/rm101_torch_module_runtime_v1.md`
- `doc/experiments/handoff/results/ottawa_torch_gated_v1.md`
- `doc/experiments/handoff/results/rm101_torch_gated_v1.md`
- `doc/experiments/handoff/results/ottawa_torch_attention_v1.md`
- `doc/experiments/handoff/results/rm101_torch_attention_v1.md`

## Preconditions

- `codex` CLI 已安装且 `codex login` 已完成
- 对应 dataset 的 torch Formal Main 已完成并记账

## Success Criteria

- 每个输出目录都存在 `final_report.md`
- 先写对应 `results/<experiment_id>.md`
- 再更新 `doc/experiments/01_result_ledger.md` 中对应 ablation row

## Failure Rule

- 不改 runtime phase 之外的实验设计
- 不覆盖 provider/model
- 失败时记 `reject`
- 只允许修改本 ticket 对应的 6 行 ablation ledger
- 不得把任何 torch ablation 改写成 main result
