# Result Ledger

| experiment_id | dataset | graph_path | phase | output_policy | llm_mode | control_mode | output_dir | accuracy | macro_f1 | keep | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ottawa_ml_pilot_v1 | Ottawa | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_ml_pilot_v1` | 1.0 | 1.0 | accept | pilot smoke (offline_stub) |
| ottawa_torch_pilot_v1 | Ottawa | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/ottawa_torch_pilot_v1` | 0.333 | 0.167 | accept | pilot smoke (offline_stub) |
| rm101_ml_pilot_v1 | RM101 | ml | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_ml_pilot_v1` | 0.25 | 0.125 | accept | pilot smoke (offline_stub) |
| rm101_torch_pilot_v1 | RM101 | torch | compiled | terminal_only | offline_stub | fixed | `artifacts/paper/rm101_torch_pilot_v1` | 0.125 | 0.028 | accept | pilot smoke (offline_stub) |
| ottawa_ml_main_v1 | Ottawa | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_ml_main_v1` |  |  |  | main/provider: stepfun/step-3.5-flash:free (pending, API parsing issue) |
| ottawa_torch_main_v1 | Ottawa | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/ottawa_torch_main_v1` |  |  |  | main/provider: stepfun/step-3.5-flash:free (pending, API parsing issue) |
| rm101_ml_main_v1 | RM101 | ml | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_ml_main_v1` |  |  |  | main/provider: stepfun/step-3.5-flash:free (pending, API parsing issue) |
| rm101_torch_main_v1 | RM101 | torch | compiled | terminal_only | provider | fixed | `artifacts/paper/rm101_torch_main_v1` |  |  |  | main/provider: stepfun/step-3.5-flash:free (pending, API parsing issue) |
