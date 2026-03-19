# Multi-Agent Merge Checklist

本文件只给 harness engineer 与 coordinator 使用，用于合并 worker 结果、验收 ledger，并决定哪些结果可以进入主表。

## Inputs

harness engineer / coordinator 只看三类输入：

- `doc/experiments/01_result_ledger.md`
- `doc/experiments/handoff/results/<experiment_id>.md`
- `artifacts/paper/<experiment_id>/`

## Merge Order

1. 先核对 `results/<experiment_id>.md`
2. 再核对 `artifacts/paper/<experiment_id>/`
3. 最后核对 `doc/experiments/01_result_ledger.md`

如果三者不一致，以顺序更靠前的原始证据为准。

## Harness Engineer Duties

harness engineer 只负责 gate，不负责选 backend winner。

### Artifact Contract Gate

以下文件缺任一项，则：

- `artifact_contract_pass=fail`
- `keep` 不得为 `accept`

硬门槛 files：

- `validated_dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `metrics.json`
- `final_report.md`

### Required Evidence Gate

以下三类证据至少要以 runtime-native artifact 或 worker report section 之一存在：

- `feature_list`
- `feature_separability_summary`
- `progress_record`

规则写死：

- 缺 `progress_record`
  - 最多记 `needs_rerun`
  - 不得进入主表
- 缺 `feature_list` 或 `feature_separability_summary`
  - Stage B row 不得参与 backend selection

### Feature Separability Gate

仅对 `ml` comparison / main / ml ablation 执行。最小要求：

- feature pipeline 非空
- 不得出现明显全零/常数/塌缩特征主导
- 必须有一个明确的 separability 结论

如果这些条件不满足：

- `feature_separability_pass=fail`
- Stage B row 不得记为 `selection_eligible=yes`

## Coordinator Duties

coordinator 只负责：

- 合并 worker 报告与 ledger
- 在通过 gate 的 Stage B row 中选择 `selected_global_best_backend`
- 更新 `doc/experiments/02_main_tables.md`

### Backend Selection Rule

只有同时满足以下条件的 Stage B row，才有资格参与比较：

- `keep=accept`
- `artifact_contract_pass=pass`
- `feature_separability_pass=pass`
- `selection_eligible=yes`

选择顺序固定为：

1. 比较 Ottawa + RM101 两条 `ml` comparison row 的 `macro_f1` 均值
2. tie-break 1：`accuracy` 均值
3. tie-break 2：更少的 incident / failed runs
4. tie-break 3：优先当前 active Codex tuple

coordinator 只能在 `doc/experiments/01_result_ledger.md` 顶部更新：

```yaml
selected_global_best_backend:
  provider: ...
  model: ...
  snapshot: ...
  selected_from_stage_b: true
```

## Acceptance Rules

### accept

- 硬门槛 artifact 完整
- required evidence 完整
- worker 报告与 ledger 一致
- 没有 provider/model 越权行为

### reject

- 运行失败
- artifact 不完整
- separability 证据明显不成立
- worker 越权改了不允许的配置
- provider/model 与 ticket 不一致

### needs_rerun

- artifact 存在但不完整
- worker 报告与 ledger 不一致
- 缺 `progress_record`
- 结果文件存在明显缺项，但尚不足以直接判 reject

## Non-Negotiable Rules

- `ml` 是 canonical diagnosis mainline
- `torch` 只做 path comparison
- `worker tool != experiment backend`
- Stage C / D 只接受 `selected_global_best_backend`
- `openrouter/free` 不能进入最终 backend 选择
- worker 不得直接更新 `doc/experiments/02_main_tables.md`
- 如果 ledger 冲突，优先保留 worker 报告，再由 harness engineer / coordinator 重写 ledger
