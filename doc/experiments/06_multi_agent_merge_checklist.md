# Multi-Agent Merge Checklist

本文件只给 coordinator 使用，用于合并 worker 结果、验收 ledger、并决定哪些结果可以进入主表。

## Inputs

coordinator 只看三类输入：

- `doc/experiments/01_result_ledger.md`
- `doc/experiments/handoff/results/<experiment_id>.md`
- `artifacts/paper/<experiment_id>/`

## Merge Order

1. 先核对 `results/<experiment_id>.md`
2. 再核对 `artifacts/paper/<experiment_id>/`
3. 最后核对 `doc/experiments/01_result_ledger.md`

如果三者不一致，以顺序更靠前的原始证据为准。

## Acceptance Rules

### accept

- `final_report.md` 存在
- 目标 artifact 存在
- worker 报告与 ledger 一致
- 没有 provider/model 越权行为

### reject

- 运行失败
- artifact 不完整
- worker 越权改了不允许的配置
- provider/model 与 ticket 不一致

### needs_rerun

- artifact 存在但不完整
- worker 报告与 ledger 不一致
- 结果文件存在明显缺项

## Coordinator Duties

- 只允许把通过验收的结果保留为 `accept`
- 只允许 coordinator 更新：
  - `doc/experiments/02_main_tables.md`
- coordinator 不替 worker 改写实验事实；只合并和纠错

## Non-Negotiable Rules

- `Formal Main` 只接受 `codex_cli / gpt-5.3-codex`
- `OpenRouter / stepfun` 只允许出现在 qualification rows
- worker 不得改 formal-main tuple
- 如果 ledger 冲突，优先保留 worker 报告，再由 coordinator 重写 ledger
