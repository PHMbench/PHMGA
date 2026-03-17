# Main Tables

## Table 1: Main Results

| experiment_id | dataset | path | phase | output_policy | llm_mode | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | offline_stub |  |  |  |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | offline_stub |  |  |  |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | offline_stub |  |  |  |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | offline_stub |  |  |  |

## Table 2: Method Ablations

| experiment_id | dataset | path | ablation_axis | setting | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_intermediate_v1 | Ottawa | ml | output_policy | include_intermediate_features |  |  |  |
| rm101_ml_intermediate_v1 | RM101 | ml | output_policy | include_intermediate_features |  |  |  |
| ottawa_torch_module_runtime_v1 | Ottawa | torch | runtime | module_runtime |  |  |  |
| rm101_torch_module_runtime_v1 | RM101 | torch | runtime | module_runtime |  |  |  |
| ottawa_torch_gated_v1 | Ottawa | torch | control | gated |  |  |  |
| rm101_torch_gated_v1 | RM101 | torch | control | gated |  |  |  |
| ottawa_torch_attention_v1 | Ottawa | torch | control | attention |  |  |  |
| rm101_torch_attention_v1 | RM101 | torch | control | attention |  |  |  |

## Table 3: Framework Ablations

| experiment_id | dataset | path | ablation_axis | setting | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_openrouter_v1 | Ottawa | ml | llm_mode | provider=openrouter |  |  |  |
| rm101_ml_openrouter_v1 | RM101 | ml | llm_mode | provider=openrouter |  |  |  |

## Source of Truth

- 所有数值必须能回指到 `doc/experiments/01_result_ledger.md`
- 所有结果目录统一位于 `artifacts/paper/<experiment_id>/`
