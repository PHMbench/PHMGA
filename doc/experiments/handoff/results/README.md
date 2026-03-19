# Worker Result Files

本目录用于保存每个 Codex CLI worker 的单次实验结果报告。

命名规则固定为：

- `doc/experiments/handoff/results/<experiment_id>.md`

这些文件是低冲突原始证据，不是主表。

正式汇总仍然写入：

- `doc/experiments/01_result_ledger.md`

worker 必须遵守固定顺序：

1. 先写本目录中的结果文件
2. 再更新 `doc/experiments/01_result_ledger.md`

coordinator 之后再根据：

- worker 结果文件
- `artifacts/paper/<experiment_id>/`
- `doc/experiments/01_result_ledger.md`

做最终验收与合并。

当前结果文件至少还要补齐：

- `progress_record`

`feature_list` 与 `feature_separability_summary` 现在默认应直接引用 runtime-native artifact；只有 `progress_record` 继续属于 provisional evidence。
