# Worker Result Template

每个 Codex CLI worker 在完成一次实验后，必须先填写一份结果报告，再更新 `doc/experiments/01_result_ledger.md`。

结果报告文件固定放在：

- `doc/experiments/handoff/results/<experiment_id>.md`

建议直接复制以下模板：

```md
# Worker Result

- worker_id:
- ticket_id:
- experiment_id:
- command:
- start_time:
- end_time:
- provider/model:
- output_dir:
- artifact_check:
- status:
- metrics_summary:
- failure_summary:
- ledger_updated:
```

## Field Rules

- `worker_id`
  - 固定写 worker 名称，例如 `pilot-owner`
- `ticket_id`
  - 固定写 handoff ticket 文件名
- `experiment_id`
  - 必须与 `artifacts/paper/<experiment_id>/` 一致
- `command`
  - 优先记录实际执行的 wrapper
- `artifact_check`
  - 至少写 `final_report.md: yes/no`
- `status`
  - 只能写：
    - `accept`
    - `reject`
    - `needs_rerun`
- `metrics_summary`
  - 成功时填写核心指标；没有指标时写 `n/a`
- `failure_summary`
  - 失败时写核心错误摘要；成功时写 `n/a`
- `ledger_updated`
  - 只能写：
    - `yes`
    - `no`

## Writeback Order

1. 先写 `results/<experiment_id>.md`
2. 再写 `doc/experiments/01_result_ledger.md`

不要反过来执行。worker 报告是低冲突原始证据，ledger 是正式汇总源。
