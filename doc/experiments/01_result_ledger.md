# Result Ledger

worker 回写规则固定为：

- worker 只允许修改自己 ticket 分配到的 row
- 先写 `doc/experiments/handoff/results/<experiment_id>.md`
- 再写本 ledger
- `keep=reject` 只用于失败运行
- `note` 必须保持对应实验类型的语义边界：
  - main
  - ablation
  - qualification

本 ledger 是正式汇总源，但不是唯一原始证据；worker 结果文件是并行证据链。

| experiment_id | dataset | graph_path | phase | output_policy | llm_mode | control_mode | output_dir | accuracy | macro_f1 | keep | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_pilot_v1 | Ottawa | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_ml_pilot_v1` | 1.0 | 1.0 | accept | pilot smoke (offline_stub) |
| ottawa_torch_pilot_v1 | Ottawa | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_torch_pilot_v1` | 0.333 | 0.167 | accept | pilot smoke (offline_stub) |
| rm101_ml_pilot_v1 | RM101 | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_ml_pilot_v1` | 0.25 | 0.125 | accept | pilot smoke (offline_stub) |
| rm101_torch_pilot_v1 | RM101 | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_torch_pilot_v1` | 0.125 | 0.028 | accept | pilot smoke (offline_stub) |
| ottawa_ml_openrouter_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_openrouter_v1` |  |  |  | provider qualification candidate: openrouter / stepfun/step-3.5-flash:free |
| ottawa_ml_codex_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_codex_v1` |  |  |  | formal-main backend sanity check: codex_cli / gpt-5.3-codex |
| rm101_ml_openrouter_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_openrouter_v1` |  |  |  | provider qualification candidate: openrouter / stepfun/step-3.5-flash:free |
| rm101_ml_codex_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_codex_v1` |  |  |  | formal-main backend sanity check: codex_cli / gpt-5.3-codex |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_main_v1` |  |  |  | formal main frozen default: codex_cli / gpt-5.3-codex |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_torch_main_v1` |  |  |  | formal main frozen default: codex_cli / gpt-5.3-codex |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_main_v1` |  |  |  | formal main frozen default: codex_cli / gpt-5.3-codex |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_torch_main_v1` |  |  |  | formal main frozen default: codex_cli / gpt-5.3-codex |
