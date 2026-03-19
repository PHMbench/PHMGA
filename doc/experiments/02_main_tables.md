# Main Tables

## Table 1: Main Results

主表按“自动诊断主线结果 + path comparison”组织；当前 canonical diagnosis backend 先固定为 `ml`，`torch` 作为比较层 path 记录。provider / backend 资格验证单独下沉到 Table 3，不参与主线定义。

| experiment_id | dataset | path | phase | output_policy | llm_mode | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | provider |  |  | canonical diagnosis mainline (codex_cli / gpt-5.3-codex) |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | provider |  |  | path comparison on the same validated DAG mainline |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | provider |  |  | canonical diagnosis mainline (codex_cli / gpt-5.3-codex) |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | provider |  |  | path comparison on the same validated DAG mainline |

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

## Table 3: Provider Qualification And Backend Sanity

| experiment_id | dataset | path | ablation_axis | setting | accuracy | macro_f1 | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_openrouter_v1 | Ottawa | ml | provider_candidate | openrouter / stepfun/step-3.5-flash:free |  |  | candidate only; not formal main default |
| ottawa_ml_codex_v1 | Ottawa | ml | backend_sanity | codex_cli / gpt-5.3-codex |  |  | sanity check for frozen formal-main tuple |
| rm101_ml_openrouter_v1 | RM101 | ml | provider_candidate | openrouter / stepfun/step-3.5-flash:free |  |  | candidate only; not formal main default |
| rm101_ml_codex_v1 | RM101 | ml | backend_sanity | codex_cli / gpt-5.3-codex |  |  | sanity check for frozen formal-main tuple |

## Source of Truth

- 所有数值必须能回指到 `doc/experiments/01_result_ledger.md`
- 所有结果目录统一位于 `artifacts/paper/<experiment_id>/`
