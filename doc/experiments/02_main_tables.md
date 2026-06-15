# Main Tables

No row enters the paper main tables unless artifact contract passed.
No pending, no no_evidence, no planner timeout, no transport failure rows.

## Table 1: Main Results

No passed run_ids yet for Table 1.

## Table 2: Best-Backend Ablations

No passed run_ids yet for Table 2.

## Table 3: Backend Comparison And Selection

Table 3 may report bounded Stage B backend-comparison evidence, but it must not report a selected global backend yet.

| experiment_id | dataset | backend | status | artifact_contract_pass | feature_separability_pass | selection_eligible | test_macro_f1 | table_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ottawa_ml_openrouter_nemotron_v3` | Ottawa | `openrouter / nvidia/nemotron-3-super-120b-a12b:free` | accept | pass | pass | yes | 0.8774661249538376 | partial Stage B evidence |
| `ottawa_ml_bigmodel_glm47_v1` | Ottawa | `bigmodel / glm-4.7-flash` | accept | pass | pass | yes | 0.7226535613558728 | partial Stage B evidence |
| `rm101_ml_openrouter_nemotron_v3` | RM101 | `openrouter / nvidia/nemotron-3-super-120b-a12b:free` | reject | pass | pass | no | 0.18337824193501234 | reject evidence; not selection-eligible |
| `rm101_ml_bigmodel_glm47_v1` | RM101 | `bigmodel / glm-4.7-flash` | reject | pass | pass | no | 0.18934628733653974 | reject evidence; not selection-eligible |

No backend has accepted Stage B evidence on both Ottawa and RM101. Therefore `selected_global_best_backend` remains `pending`, and Stage C/D rows remain locked.

## Source of Truth

- 所有最终表格行都必须能回指到 `doc/experiments/01_result_ledger.md`
- 所有最终表格行都必须能回指到对应 `result_md`
- 所有最终表格行都必须能回指到对应 `artifact_dir`
