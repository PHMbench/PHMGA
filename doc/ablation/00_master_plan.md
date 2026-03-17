# Ablation Master Plan

## Scope

- 对比 `compiled`、`module_runtime`、`learnable_control`
- 记录 WaveFilters family 的引入效果
- 记录 gate / attention 对真实 Ottawa 与 RM101 的影响

## Fixed Baselines

- `offline_stub`
- `output_policy=terminal_only`
- `phase=compiled`
- 正式入口统一用 `main.py +runs=... runtime.output_dir=...`

## Priority

1. `compiled` vs `module_runtime`
2. fixed vs gated
3. gated vs attention
4. WaveFilters family 内部对比

## Ledger Rule

- 消融结果统一登记到 `doc/experiments/01_result_ledger.md`
- 主表不写在这里，统一维护在 `doc/experiments/02_main_tables.md`
