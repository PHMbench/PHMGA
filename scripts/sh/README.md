# Shell Wrappers

`scripts/sh/` 只提供对常用实验命令的便利封装。

正式入口仍然是仓库根目录 [main.py](/home/user/LQ/B_Signal/PHMGA/main.py)，权威实验说明与结果管理位于：

- [doc/experiments/00_manual_runbook.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/00_manual_runbook.md)
- [doc/experiments/01_result_ledger.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/01_result_ledger.md)
- [doc/experiments/02_main_tables.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/02_main_tables.md)
- [doc/experiments/04_codex_cli_handoff.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/04_codex_cli_handoff.md)
- [doc/experiments/05_worker_result_template.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/05_worker_result_template.md)
- [doc/experiments/06_multi_agent_merge_checklist.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/06_multi_agent_merge_checklist.md)

## Role

- 这些脚本只是 `python main.py +runs=...` 的薄包装。
- `scripts/sh/_common.sh` 是共享 helper，不是实验 wrapper。
- Pilot wrapper 会先跑 `runtime.action=preflight`，并强制 `llm.mode=offline_stub`。
- Main wrapper 会检查对应 pilot 的 `final_report.md`，并默认继承 Formal Main 已冻结的 Codex tuple。
- Ablation wrapper 只封装 runbook 中已经写死的 override。

## Current Wrapper Groups

- `pilot/`
- `main/`
- `ablation/output_policy/`
- `ablation/runtime/`
- `ablation/control/gated/`
- `ablation/control/attention/`
- `ablation/provider/`

其中：

- `pilot/`
  - smoke only
  - 总是显式传入 `llm.mode=offline_stub`
- `main/`
  - 默认继承 preset 内冻结的 Formal Main tuple：
    - `provider=codex_cli`
    - `model=gpt-5.3-codex`
  - 可通过环境变量临时覆盖：
    - `PHMGA_LLM_PROVIDER`
    - `PHMGA_LLM_MODEL`
  - 正式 evidence run 默认不应覆盖；只有诊断或候选对比时才覆盖
  - 例如：

```bash
./scripts/sh/main/01_ottawa_ml_main.sh
```

- `ablation/provider/`
  - 不是“正式默认 main”
  - 用于 provider qualification 或 frozen main backend sanity check
  - 当前已提供：
    - `*openrouter.sh` -> 默认 `stepfun/step-3.5-flash:free`
    - `*codex.sh` -> 默认 `gpt-5.3-codex`，作为 Formal Main frozen tuple 的 sanity check
  - 这些 wrapper 会锁住 provider，仅允许通过 `PHMGA_LLM_MODEL` 覆盖模型名

当前不再通过 shell wrapper 暴露 WaveFilters 实验；WaveFilters 仍以研究计划形式保留在：

- [doc/ablation/02_wavefilters_ablation.md](/home/user/LQ/B_Signal/PHMGA/doc/ablation/02_wavefilters_ablation.md)

## Usage

首次使用：

```bash
./scripts/sh/SETUP.sh
```

执行单个 wrapper：

```bash
./scripts/sh/pilot/01_ottawa_ml_pilot.sh
./scripts/sh/main/02_ottawa_torch_main.sh
./scripts/sh/ablation/control/gated/ottawa_torch_gated.sh
./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh
./scripts/sh/ablation/provider/ottawa_ml_codex.sh
```

查看当前实际 wrapper 数量时，应把 `scripts/sh/_common.sh` 排除在外；它只是共享 shell helper。

执行顺序、实验命名和 ledger 规则以 runbook 为准，不以本目录文档为准。

如果这些 wrapper 交给其他 Codex CLI worker 执行，必须同时遵守：

- `doc/experiments/04_codex_cli_handoff.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`
- `doc/experiments/handoff/*.md`
