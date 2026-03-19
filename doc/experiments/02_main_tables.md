# Main Tables

## Table 1: Main Results

主表只表述自动诊断主线结果与 path comparison。  
当前 canonical diagnosis mainline 固定为 `ml`；`torch` 作为同一 selected backend 下的 path comparison。

| experiment_id | dataset | path | phase | output_policy | llm_mode | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | provider |  |  | canonical diagnosis mainline on selected_global_best_backend |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | provider |  |  | path comparison on the same selected_global_best_backend |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | provider |  |  | canonical diagnosis mainline on selected_global_best_backend |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | provider |  |  | path comparison on the same selected_global_best_backend |

## Table 2: Best-Backend Ablations

Table 2 只记录 selected_global_best_backend 的消融，不按 backend 复制矩阵。

| experiment_id | dataset | path | ablation_axis | setting | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_intermediate_v1 | Ottawa | ml | output_policy | include_intermediate_features |  |  | ablation on selected_global_best_backend |
| rm101_ml_intermediate_v1 | RM101 | ml | output_policy | include_intermediate_features |  |  | ablation on selected_global_best_backend |
| ottawa_torch_module_runtime_v1 | Ottawa | torch | runtime | module_runtime |  |  | ablation on selected_global_best_backend |
| rm101_torch_module_runtime_v1 | RM101 | torch | runtime | module_runtime |  |  | ablation on selected_global_best_backend |
| ottawa_torch_gated_v1 | Ottawa | torch | control | gated |  |  | ablation on selected_global_best_backend |
| rm101_torch_gated_v1 | RM101 | torch | control | gated |  |  | ablation on selected_global_best_backend |
| ottawa_torch_attention_v1 | Ottawa | torch | control | attention |  |  | ablation on selected_global_best_backend |
| rm101_torch_attention_v1 | RM101 | torch | control | attention |  |  | ablation on selected_global_best_backend |

## Table 3: Backend Comparison And Selection

Table 3 只记录 Stage B 的 active comparison set。  
只有 `selection_eligible=yes` 的 row 才参与 backend 选择。

| experiment_id | dataset | path | ablation_axis | setting | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_codex_v1 | Ottawa | ml | backend_comparison | codex_cli / gpt-5.3-codex |  |  | current active codex comparison tuple |
| ottawa_ml_openrouter_v1 | Ottawa | ml | backend_comparison | openrouter / stepfun/step-3.5-flash:free |  |  | current active OpenRouter comparison tuple |
| rm101_ml_codex_v1 | RM101 | ml | backend_comparison | codex_cli / gpt-5.3-codex |  |  | current active codex comparison tuple |
| rm101_ml_openrouter_v1 | RM101 | ml | backend_comparison | openrouter / stepfun/step-3.5-flash:free |  |  | current active OpenRouter comparison tuple |

### Selection Rule

- only rows with `keep=accept`, `artifact_contract_pass=pass`, `feature_separability_pass=pass`, and `selection_eligible=yes` may participate
- primary rank: mean `macro_f1` over Ottawa + RM101 on canonical `ml`
- tie-break 1: mean `accuracy`
- tie-break 2: fewer incidents / failed runs
- final tie-break: prefer current active Codex tuple

## Source of Truth

- 所有数值必须能回指到 `doc/experiments/01_result_ledger.md`
- 所有结果目录统一位于 `artifacts/paper/<experiment_id>/`
