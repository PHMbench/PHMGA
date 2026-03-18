# Shell Wrappers

`scripts/sh/` 只提供对常用实验命令的便利封装。

正式入口仍然是仓库根目录 [main.py](/home/user/LQ/B_Signal/PHMGA/main.py)，权威实验说明与结果管理位于：

- [doc/experiments/00_manual_runbook.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/00_manual_runbook.md)
- [doc/experiments/01_result_ledger.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/01_result_ledger.md)
- [doc/experiments/02_main_tables.md](/home/user/LQ/B_Signal/PHMGA/doc/experiments/02_main_tables.md)

## Role

- 这些脚本只是 `python main.py +runs=...` 的薄包装。
- Pilot wrapper 会先跑 `runtime.action=preflight`。
- Main wrapper 会检查对应 pilot 的 `final_report.md`。
- Ablation wrapper 只封装 runbook 中已经写死的 override。

## Current Wrapper Groups

- `pilot/`
- `main/`
- `ablation/output_policy/`
- `ablation/runtime/`
- `ablation/control/gated/`
- `ablation/control/attention/`
- `ablation/provider/`

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
```

执行顺序、实验命名和 ledger 规则以 runbook 为准，不以本目录文档为准。
