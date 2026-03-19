# Worker Result Template

每个 Codex CLI worker 在完成一次实验后，必须先填写一份结果报告，再更新 `doc/experiments/01_result_ledger.md`。

结果报告文件固定放在：

- `doc/experiments/handoff/results/<experiment_id>.md`

建议直接复制以下模板：

```md
# Worker Result

- worker_id:
- ticket_id:
- run_type:
- experiment_id:
- command:
- start_time:
- end_time:
- provider/model:
- output_dir:
- artifact_contract_pass:
- feature_separability_pass:
- status:
- ledger_updated:

## Artifact Checklist

- validated_dag.json:
- compiled_dag_manifest.json:
- feature_pipeline.json:
- feature_list.json:
- feature_separability_summary.json:
- artifact_index.json:
- metrics.json:
- final_report.md:

## Required Evidence

### feature_list

### feature_separability_summary

### progress_record

## Metrics Summary

## Failure Summary

## Notes
```

## Field Rules

- `worker_id`
  - 固定写 worker 名称，例如 `pilot-owner`
- `ticket_id`
  - 固定写 handoff ticket 文件名
- `run_type`
  - 只能写：
    - `pilot`
    - `backend_comparison`
    - `formal_main`
    - `ablation`
- `experiment_id`
  - 必须与 `artifacts/paper/<experiment_id>/` 一致
- `command`
  - 优先记录实际执行的 wrapper；如果使用了 env override，要原样写出
- `provider/model`
  - 固定写实际实验 backend tuple，而不是 worker tool
- `artifact_contract_pass`
  - worker 先写：
    - `pass`
    - `fail`
    - `pending_harness_review`
- `feature_separability_pass`
  - 只对 `ml` comparison / main / ml ablation 有意义；其余可写 `n/a`
- `status`
  - 只能写：
    - `accept`
    - `reject`
    - `needs_rerun`
- `ledger_updated`
  - 只能写：
    - `yes`
    - `no`

## Evidence Rules

- `Artifact Checklist`
  - 所有硬门槛 artifact 都要逐项写存在性
- `feature_list`
  - 直接引用 `feature_list.json`
- `feature_separability_summary`
  - 直接引用 `feature_separability_summary.json`
- `progress_record`
  - 优先引用 `progress.json`
  - 如果没有 runtime-native 文件，必须在这里写出阶段进展记录

## Writeback Order

1. 先写 `results/<experiment_id>.md`
2. 再写 `doc/experiments/01_result_ledger.md`

不要反过来执行。worker 报告是低冲突原始证据，ledger 是正式汇总源。
