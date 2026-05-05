# Result Ledger

本 ledger 只记录 formal paper runs 的状态真值。proving lane 不写入这里；对应证明链状态请回看：

- `doc/experiments/00_manual_runbook.md`
- `artifacts/proving/*`

worker 回写规则固定为：

- worker 只允许修改自己 ticket 分配到的 row
- 先写 `doc/experiments/handoff/results/<experiment_id>.md`
- 再写本 ledger
- `keep=accept` 只在 formal row 真正通过后使用
- `keep=reject` 只用于确认失败的 formal row
- `pending` formal rows 可以存在，但不能被 `doc/experiments/02_main_tables.md` 引用
- `artifact_contract_pass`、`feature_separability_pass`、`selection_eligible` 由 harness engineer 或 coordinator 复核，不由普通 worker 自行裁定为最终事实

```yaml
candidate_registry:
  codex:
    - provider: codex_cli
      model: gpt-5.4
      snapshot: null
    - provider: codex_cli
      model: gpt-5.2
      snapshot: null
    - provider: codex_cli
      model: gpt-5.3-codex
      snapshot: null
  openrouter:
    - provider: openrouter
      model: z-ai/glm-4.5-air:free
      snapshot: null
    - provider: openrouter
      model: nvidia/nemotron-3-super-120b-a12b:free
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
  bigmodel:
    - provider: bigmodel
      model: glm-4.7-flash
      snapshot: null

active_stage_b_set:
  codex:
    provider: codex_cli
    model: gpt-5.3-codex
    snapshot: null
  openrouter:
    provider: openrouter
    model: z-ai/glm-4.5-air:free
    snapshot: null
  bigmodel:
    provider: bigmodel
    model: glm-4.7-flash
    snapshot: null

selected_global_best_backend:
  provider: pending
  model: pending
  snapshot: null
  status: pending
  selected_from_stage_b: false
  selection_basis: mean macro_f1 over Ottawa + RM101 canonical ml mainline
```

Stage B row 只有在以下条件同时满足时，才能记为 `selection_eligible=yes`：

- `keep=accept`
- `artifact_contract_pass=pass`
- `feature_separability_pass=pass`

| experiment_id | preset_name | dataset | graph_path | run_type | artifact_dir | result_md | artifact_contract_pass | feature_separability_pass | selection_eligible | accuracy | macro_f1 | keep | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ottawa_ml_codex_v3` | `ottawa_ml_codex_v3` | Ottawa | ml | backend_comparison | `artifacts/paper/ottawa_ml_codex_v3` | `doc/experiments/handoff/results/ottawa_ml_codex_v3.md (expected)` | pending | pending | no |  |  | pending | backend comparison candidate: `codex_cli / gpt-5.3-codex`; active v3 comparison row with DAG depth target `3-8` |
| `ottawa_ml_openrouter_glm_v2` | `ottawa_ml_openrouter_glm_v2` | Ottawa | ml | backend_comparison | `artifacts/paper/ottawa_ml_openrouter_glm_v2` | `doc/experiments/handoff/results/ottawa_ml_openrouter_glm_v2.md` | fail | fail | no |  |  | reject | 2026-05-04 rerun rejected: preflight passed, planner repair normalized a two-step plan, then OpenRouter free upstream returned HTTP 429 for `z-ai/glm-4.5-air:free`; no validated DAG or ML artifacts |
| `ottawa_ml_openrouter_nemotron_v3` | `ottawa_ml_openrouter_nemotron_v3` | Ottawa | ml | backend_comparison | `artifacts/paper/ottawa_ml_openrouter_nemotron_v3_qualityfix1` | `doc/experiments/handoff/results/ottawa_ml_openrouter_nemotron_v3.md` | pass | pass | yes | 0.8770491803278688 | 0.8774661249538376 | accept | 2026-05-04 qualityfix1 emitted a complete artifact bundle with `nvidia/nemotron-3-super-120b-a12b:free`; artifact contract and feature separability passed, final report was provider-authored, and workflow reached finish through deterministic quality fallback after invalid reflection output |
| `ottawa_ml_bigmodel_glm47_v1` | `ottawa_ml_bigmodel_glm47_v1` | Ottawa | ml | backend_comparison | `artifacts/paper/ottawa_ml_bigmodel_glm47_v1_qualityfix2` | `doc/experiments/handoff/results/ottawa_ml_bigmodel_glm47_v1.md` | pass | pass | yes | 0.7254098360655737 | 0.7226535613558728 | accept | 2026-05-04 qualityfix2 emitted complete artifact bundle with `glm-4.7-flash`; DAG quality finish_candidate, feature separability pass, and test macro_f1 0.7226535613558728; final report used deterministic provider fallback after report-stage provider error |
| `rm101_ml_codex_v3` | `rm101_ml_codex_v3` | RM101 | ml | backend_comparison | `artifacts/paper/rm101_ml_codex_v3` | `doc/experiments/handoff/results/rm101_ml_codex_v3.md (expected)` | pending | pending | no |  |  | pending | backend comparison candidate: `codex_cli / gpt-5.3-codex`; active v3 comparison row with DAG depth target `3-8` |
| `rm101_ml_openrouter_glm_v2` | `rm101_ml_openrouter_glm_v2` | RM101 | ml | backend_comparison | `artifacts/paper/rm101_ml_openrouter_glm_v2` | `doc/experiments/handoff/results/rm101_ml_openrouter_glm_v2.md` | pending | pending | no |  |  | pending | backend comparison candidate: `openrouter / z-ai/glm-4.5-air:free`; active free-model comparison row with DAG depth target `3-8` |
| `rm101_ml_openrouter_nemotron_v3` | `rm101_ml_openrouter_nemotron_v3` | RM101 | ml | backend_comparison | `artifacts/paper/rm101_ml_openrouter_nemotron_v3_qualityfix1` | `doc/experiments/handoff/results/rm101_ml_openrouter_nemotron_v3.md` | pass | pass | no | 0.2371657754010695 | 0.18337824193501234 | reject | 2026-05-04 qualityfix1 emitted a complete reject-evidence bundle with `nvidia/nemotron-3-super-120b-a12b:free`; artifact/feature gates passed but workflow_exit shows max_iterations reached before finish with last_reflection_decision=need_replan, so it is not selection-eligible |
| `rm101_ml_bigmodel_glm47_v1` | `rm101_ml_bigmodel_glm47_v1` | RM101 | ml | backend_comparison | `artifacts/paper/rm101_ml_bigmodel_glm47_v1_qualityfix7` | `doc/experiments/handoff/results/rm101_ml_bigmodel_glm47_v1.md` | pass | pass | no | 0.2429144385026738 | 0.18934628733653974 | reject | 2026-05-04 qualityfix7 emitted a complete reject-evidence bundle with deterministic provider fallbacks; workflow_exit shows max_iterations reached before finish with last_reflection_decision=need_patch, so it is not selection-eligible despite artifact/feature gates passing |
| `ottawa_ml_main_v1` | `ottawa_ml` | Ottawa | ml | main | `artifacts/paper/ottawa_ml_main_v1` | `doc/experiments/handoff/results/ottawa_ml_main_v1.md (missing)` | pending | pending | n/a |  |  | pending | formal main using `selected_global_best_backend` on canonical diagnosis mainline |
| `ottawa_torch_main_v1` | `ottawa_torch` | Ottawa | torch | main | `artifacts/paper/ottawa_torch_main_v1` | `doc/experiments/handoff/results/ottawa_torch_main_v1.md (missing)` | pending | n/a | n/a |  |  | pending | path comparison using `selected_global_best_backend` |
| `rm101_ml_main_v1` | `rm101_ml` | RM101 | ml | main | `artifacts/paper/rm101_ml_main_v1` | `doc/experiments/handoff/results/rm101_ml_main_v1.md (missing)` | pending | pending | n/a |  |  | pending | formal main using `selected_global_best_backend` on canonical diagnosis mainline |
| `rm101_torch_main_v1` | `rm101_torch` | RM101 | torch | main | `artifacts/paper/rm101_torch_main_v1` | `doc/experiments/handoff/results/rm101_torch_main_v1.md (missing)` | pending | n/a | n/a |  |  | pending | path comparison using `selected_global_best_backend` |
| `ottawa_ml_intermediate_v1` | `ottawa_ml` | Ottawa | ml | ablation | `artifacts/paper/ottawa_ml_intermediate_v1` | `doc/experiments/handoff/results/ottawa_ml_intermediate_v1.md (missing)` | pending | pending | n/a |  |  | pending | ablation on `selected_global_best_backend`: output_policy |
| `rm101_ml_intermediate_v1` | `rm101_ml` | RM101 | ml | ablation | `artifacts/paper/rm101_ml_intermediate_v1` | `doc/experiments/handoff/results/rm101_ml_intermediate_v1.md (missing)` | pending | pending | n/a |  |  | pending | ablation on `selected_global_best_backend`: output_policy |
| `ottawa_torch_module_runtime_v1` | `ottawa_torch` | Ottawa | torch | ablation | `artifacts/paper/ottawa_torch_module_runtime_v1` | `doc/experiments/handoff/results/ottawa_torch_module_runtime_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: module_runtime |
| `rm101_torch_module_runtime_v1` | `rm101_torch` | RM101 | torch | ablation | `artifacts/paper/rm101_torch_module_runtime_v1` | `doc/experiments/handoff/results/rm101_torch_module_runtime_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: module_runtime |
| `ottawa_torch_gated_v1` | `ottawa_torch` | Ottawa | torch | ablation | `artifacts/paper/ottawa_torch_gated_v1` | `doc/experiments/handoff/results/ottawa_torch_gated_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: gated control |
| `rm101_torch_gated_v1` | `rm101_torch` | RM101 | torch | ablation | `artifacts/paper/rm101_torch_gated_v1` | `doc/experiments/handoff/results/rm101_torch_gated_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: gated control |
| `ottawa_torch_attention_v1` | `ottawa_torch` | Ottawa | torch | ablation | `artifacts/paper/ottawa_torch_attention_v1` | `doc/experiments/handoff/results/ottawa_torch_attention_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: attention control |
| `rm101_torch_attention_v1` | `rm101_torch` | RM101 | torch | ablation | `artifacts/paper/rm101_torch_attention_v1` | `doc/experiments/handoff/results/rm101_torch_attention_v1.md (missing)` | pending | n/a | n/a |  |  | pending | ablation on `selected_global_best_backend`: attention control |

## Historical Comparison Incidents Retained Outside Formal Selection

以下历史 comparison rows 保留为并行证据链与事故记录，不参与当前 formal selection，也不会进入 runbook 总矩阵或 main tables：

- `ottawa_ml_codex_v1` → `doc/experiments/handoff/results/ottawa_ml_codex_v1.md`
- `ottawa_ml_codex_v2` → `doc/experiments/handoff/results/ottawa_ml_codex_v2.md`
- `ottawa_ml_openrouter_glm_v1` → no worker result file; superseded by `ottawa_ml_openrouter_glm_v2`
- `ottawa_ml_openrouter_glm_v2` → `doc/experiments/handoff/results/ottawa_ml_openrouter_glm_v2.md`
- `ottawa_ml_openrouter_nemotron_v3` → `doc/experiments/handoff/results/ottawa_ml_openrouter_nemotron_v3.md`
- `ottawa_ml_openrouter_v1` → `doc/experiments/handoff/results/ottawa_ml_openrouter_v1.md`
- `rm101_ml_codex_v1` → `doc/experiments/handoff/results/rm101_ml_codex_v1.md`
- `rm101_ml_codex_v2` → `doc/experiments/handoff/results/rm101_ml_codex_v2.md`
- `rm101_ml_openrouter_glm_v1` → no worker result file; superseded by `rm101_ml_openrouter_glm_v2`
- `rm101_ml_openrouter_glm_v2` → `doc/experiments/handoff/results/rm101_ml_openrouter_glm_v2.md`
- `rm101_ml_openrouter_nemotron_v3` → `doc/experiments/handoff/results/rm101_ml_openrouter_nemotron_v3.md`
- `rm101_ml_openrouter_v1` → `doc/experiments/handoff/results/rm101_ml_openrouter_v1.md`
- `doc/experiments/incidents/03_openrouter_api_analysis.md`
