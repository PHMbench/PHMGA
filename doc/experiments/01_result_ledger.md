# Result Ledger

worker 回写规则固定为：

- worker 只允许修改自己 ticket 分配到的 row
- 先写 `doc/experiments/handoff/results/<experiment_id>.md`
- 再写本 ledger
- `keep=reject` 只用于失败运行
- `note` 必须保持对应实验类型的语义边界：
  - pilot
  - backend_comparison
  - main
  - ablation
- `artifact_contract_pass`、`feature_separability_pass`、`selection_eligible` 由 harness engineer 或 coordinator 复核，不由普通 worker 自行裁定为最终事实

本 ledger 是正式汇总源，但不是唯一原始证据；worker 结果文件是并行证据链。

```yaml
codex_candidate_registry:
  - provider: codex_cli
    model: gpt-5.4
    snapshot: null
  - provider: codex_cli
    model: gpt-5.2
    snapshot: null
  - provider: codex_cli
    model: gpt-5.3-codex
    snapshot: null

openrouter_candidate_registry:
  - provider: openrouter
    model: z-ai/glm-4.5-air:free
    snapshot: null
  - provider: openrouter
    model: stepfun/step-3.5-flash:free
    snapshot: null
  - provider: openrouter
    model: google/gemini-2.0-flash-exp
    snapshot: null
  - provider: openrouter
    model: google/gemini-2.5-pro
    snapshot: null
  - provider: openrouter
    model: openrouter/free
    snapshot: null
    selection_eligible: false

active_stage_b_set:
  codex:
    provider: codex_cli
    model: gpt-5.3-codex
    snapshot: null
  openrouter:
    provider: openrouter
    model: z-ai/glm-4.5-air:free
    snapshot: null

selected_global_best_backend:
  provider: pending
  model: pending
  snapshot: null
  status: pending
  selected_from_stage_b: false
  selection_basis: mean macro_f1 over Ottawa + RM101 canonical ml mainline
```

Stage B 只有在以下条件同时满足时，row 才能记为 `selection_eligible=yes`：

- `keep=accept`
- `artifact_contract_pass=pass`
- `feature_separability_pass=pass`

| experiment_id | dataset | graph_path | phase | output_policy | llm_mode | control_mode | output_dir | artifact_contract_pass | feature_separability_pass | selection_eligible | accuracy | macro_f1 | keep | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_pilot_v1 | Ottawa | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_ml_pilot_v1` | n/a | n/a | n/a | 1.0 | 1.0 | accept | pilot smoke (offline_stub) |
| ottawa_torch_pilot_v1 | Ottawa | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_torch_pilot_v1` | n/a | n/a | n/a | 0.333 | 0.167 | accept | pilot smoke (offline_stub) |
| rm101_ml_pilot_v1 | RM101 | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_ml_pilot_v1` | n/a | n/a | n/a | 0.25 | 0.125 | accept | pilot smoke (offline_stub) |
| rm101_torch_pilot_v1 | RM101 | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_torch_pilot_v1` | n/a | n/a | n/a | 0.125 | 0.028 | accept | pilot smoke (offline_stub) |
| ottawa_ml_codex_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_codex_v1` | fail | fail | no |  |  | reject | backend comparison candidate: codex_cli / gpt-5.3-codex; planner blocked in `codex exec`, no artifacts emitted after extended wait |
| ottawa_ml_openrouter_glm_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_openrouter_glm_v1` | pending | pending | no |  |  |  | backend comparison candidate: openrouter / z-ai/glm-4.5-air:free; current active OpenRouter tuple for this selection round |
| rm101_ml_codex_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_codex_v1` | fail | fail | no |  |  | reject | backend comparison candidate: codex_cli / gpt-5.3-codex; planner failed to return within a bounded 180s Stage B window |
| rm101_ml_openrouter_glm_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_openrouter_glm_v1` | pending | pending | no |  |  |  | backend comparison candidate: openrouter / z-ai/glm-4.5-air:free; current active OpenRouter tuple for this selection round |
| ottawa_ml_openrouter_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_openrouter_v1` | fail | fail | no |  |  | reject | historical comparison failure: openrouter / stepfun/step-3.5-flash:free; planner output was not normalizable into `StepPlan` after repair; not in current active Stage B set |
| rm101_ml_openrouter_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_openrouter_v1` | fail | fail | no |  |  | reject | historical comparison failure: openrouter / stepfun/step-3.5-flash:free; planner output was not normalizable into `StepPlan` after repair; not in current active Stage B set |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_main_v1` | pending | pending | n/a |  |  |  | formal main using selected_global_best_backend on canonical diagnosis mainline |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_torch_main_v1` | pending | n/a | n/a |  |  |  | path comparison using selected_global_best_backend |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_main_v1` | pending | pending | n/a |  |  |  | formal main using selected_global_best_backend on canonical diagnosis mainline |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_torch_main_v1` | pending | n/a | n/a |  |  |  | path comparison using selected_global_best_backend |
| ottawa_ml_intermediate_v1 | Ottawa | ml | compiled | include_intermediate_features | provider | fixed | `artifacts/paper/ottawa_ml_intermediate_v1` | pending | pending | n/a |  |  |  | ablation on selected_global_best_backend: output_policy |
| rm101_ml_intermediate_v1 | RM101 | ml | compiled | include_intermediate_features | provider | fixed | `artifacts/paper/rm101_ml_intermediate_v1` | pending | pending | n/a |  |  |  | ablation on selected_global_best_backend: output_policy |
| ottawa_torch_module_runtime_v1 | Ottawa | torch | module_runtime | terminal_only | provider | fixed | `artifacts/paper/ottawa_torch_module_runtime_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: module_runtime |
| rm101_torch_module_runtime_v1 | RM101 | torch | module_runtime | terminal_only | provider | fixed | `artifacts/paper/rm101_torch_module_runtime_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: module_runtime |
| ottawa_torch_gated_v1 | Ottawa | torch | learnable_control | terminal_only | provider | gated | `artifacts/paper/ottawa_torch_gated_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: gated control |
| rm101_torch_gated_v1 | RM101 | torch | learnable_control | terminal_only | provider | gated | `artifacts/paper/rm101_torch_gated_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: gated control |
| ottawa_torch_attention_v1 | Ottawa | torch | learnable_control | terminal_only | provider | attention | `artifacts/paper/ottawa_torch_attention_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: attention control |
| rm101_torch_attention_v1 | RM101 | torch | learnable_control | terminal_only | provider | attention | `artifacts/paper/rm101_torch_attention_v1` | pending | n/a | n/a |  |  |  | ablation on selected_global_best_backend: attention control |
