# Codex CLI Worker Handoff

本文件定义把 PHMGA 实验交给其他 Codex CLI worker 时的唯一交接合同。

## Authority

worker 只依赖以下材料：

- `README.md`
- `doc/experiments/00_manual_runbook.md`
- `doc/experiments/01_result_ledger.md`
- `scripts/sh/README.md`
- 本文件

如果这些材料之间出现冲突，优先级固定为：

1. `doc/experiments/00_manual_runbook.md`
2. `doc/experiments/01_result_ledger.md`
3. `scripts/sh/README.md`
4. 本文件
5. `README.md`

## Worker Contract

- 其他 agent 统一视为 `Codex CLI worker`
- worker 负责：
  - 读取 ticket
  - 执行 shell wrapper
  - 检查 `final_report.md` 和目标 artifact
  - 先写 worker 结果报告
  - 再更新 `doc/experiments/01_result_ledger.md`
- worker 不负责：
  - 改代码
  - 不得改 provider 默认
  - 改 `config/runs/*.yaml`
  - 改 experiment matrix
  - 改 main table 语义
  - 调参
  - 自行更换模型

## Fixed Experiment Facts

- `Formal Main = codex_cli / gpt-5.3-codex`
- `OpenRouter / stepfun/step-3.5-flash:free = qualification candidate only`
- `pilot = offline_stub`
- 正式入口是根目录 `main.py`
- shell wrapper 只是薄封装；优先使用 wrapper，不手写新命令
- worker 结果文件固定写到：
  - `doc/experiments/handoff/results/<experiment_id>.md`

## Environment

- 所有 worker 在同一台机器、同一个仓库工作区、同一个 `.venv` 中运行
- Formal Main 与 method ablation 需要：
  - `codex` CLI 已安装
  - `codex login` 已完成
- OpenRouter qualification 额外需要：
  - `OPENROUTER_API_KEY`
- 输出目录固定为：
  - `artifacts/paper/<experiment_id>/`

推荐启动前检查：

```bash
cd /home/user/LQ/B_Signal/PHMGA
./scripts/sh/SETUP.sh
which codex
```

qualification worker 额外检查：

```bash
test -n "${OPENROUTER_API_KEY:-}"
```

## Failure Rules

- 运行成功后：
  - 检查目标 `final_report.md`
  - 先写 `results/<experiment_id>.md`
  - 再更新 `doc/experiments/01_result_ledger.md`
- 运行失败后：
  - 不删 artifact
  - 不换 provider/model
  - 先写 `results/<experiment_id>.md`
  - 在 ledger 中登记 `keep=reject`
  - `note` 写明失败原因
- worker 不得直接填写 `doc/experiments/02_main_tables.md`
- 如果 ledger 写回发生冲突，先保留 worker 报告，由 coordinator 重新合并 ledger

## Worker Tickets

- `doc/experiments/handoff/01_pilot_owner.md`
- `doc/experiments/handoff/02_codex_main_owner_ottawa.md`
- `doc/experiments/handoff/03_codex_main_owner_rm101.md`
- `doc/experiments/handoff/04_ablation_owner_ml.md`
- `doc/experiments/handoff/05_ablation_owner_torch.md`
- `doc/experiments/handoff/06_qualification_owner.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`

每个 ticket 都已经写死：

- 允许执行的 wrapper
- 对应 `experiment_id`
- 对应 `results/<experiment_id>.md`
- 必须更新的 ledger row
- 成功标准
- 失败处理规则
