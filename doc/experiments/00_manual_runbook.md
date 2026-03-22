# Manual Experiment Runbook

`scripts/sh/` 只提供对本 runbook 常用命令的薄包装；正式、权威的实验命令、阶段定义、结果位置和当前状态仍以本文件为准。

如果要把实验交给其他 Codex CLI worker 执行，统一使用：

- `doc/experiments/04_execution_protocol.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/handoff/*.md`

## Experiment Matrix

这是唯一的首屏总矩阵，用来直接回答四个问题：做哪些实验、跑什么命令、结果在哪、当前状态如何。

| layer | stage | preset_name | experiment_id | dataset | provider | model | workflow_mode | command | artifact_dir | result_md | paper_target | current_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `M0` | `proving` | `ottawa_ml_codex_proving` | `ottawa_ml_codex_proving` | Ottawa | `codex_cli` | `gpt-5.3-codex` | `supervisor_proving` | `python main.py +runs=ottawa_ml_codex_proving` | `artifacts/proving/ottawa_ml_codex_proving` | `n/a (proving lane; use artifact bundle)` | engineering qualification | `pass` |
| `M0` | `proving` | `ottawa_ml_openrouter_glm_proving` | `ottawa_ml_openrouter_glm_proving` | Ottawa | `openrouter` | `z-ai/glm-4.5-air:free` | `supervisor_proving` | `python main.py +runs=ottawa_ml_openrouter_glm_proving` | `artifacts/proving/ottawa_ml_openrouter_glm_proving` | `n/a (proving lane; use artifact bundle)` | engineering qualification | `pass_with_local_incident` |
| `M0` | `simple_qualification` | `ottawa_ml_codex_simple` | `ottawa_ml_codex_simple` | Ottawa | `codex_cli` | `gpt-5.3-codex` | `simple_fullchain` | `python main.py runtime.action=preflight +runs=ottawa_ml_codex_simple && python main.py +runs=ottawa_ml_codex_simple` | `artifacts/simple/ottawa_ml_codex_simple` | `doc/experiments/handoff/results/ottawa_ml_codex_simple.md` | engineering qualification | `pass_with_local_incident` |
| `M0` | `simple_qualification` | `rm101_ml_codex_simple` | `rm101_ml_codex_simple` | RM101 | `codex_cli` | `gpt-5.3-codex` | `simple_fullchain` | `python main.py runtime.action=preflight +runs=rm101_ml_codex_simple && python main.py +runs=rm101_ml_codex_simple` | `artifacts/simple/rm101_ml_codex_simple` | `doc/experiments/handoff/results/rm101_ml_codex_simple.md` | engineering qualification | `fail` |
| `M2` | `stage_b` | `ottawa_ml_codex_v3` | `ottawa_ml_codex_v3` | Ottawa | `codex_cli` | `gpt-5.3-codex` | `rich` | `python main.py runtime.action=preflight +runs=ottawa_ml_codex_v3 && python main.py +runs=ottawa_ml_codex_v3` | `artifacts/paper/ottawa_ml_codex_v3` | `doc/experiments/handoff/results/ottawa_ml_codex_v3.md` | Table 3 comparison gate | `pending` |
| `M2` | `stage_b` | `ottawa_ml_openrouter_nemotron_v3` | `ottawa_ml_openrouter_nemotron_v3` | Ottawa | `openrouter` | `nvidia/nemotron-3-super-120b-a12b:free` | `rich` | `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py runtime.action=preflight +runs=ottawa_ml_openrouter_nemotron_v3 && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py +runs=ottawa_ml_openrouter_nemotron_v3` | `artifacts/paper/ottawa_ml_openrouter_nemotron_v3` | `doc/experiments/handoff/results/ottawa_ml_openrouter_nemotron_v3.md` | Table 3 comparison gate | `pending` |
| `M2` | `stage_b` | `rm101_ml_codex_v3` | `rm101_ml_codex_v3` | RM101 | `codex_cli` | `gpt-5.3-codex` | `rich` | `python main.py runtime.action=preflight +runs=rm101_ml_codex_v3 && python main.py +runs=rm101_ml_codex_v3` | `artifacts/paper/rm101_ml_codex_v3` | `doc/experiments/handoff/results/rm101_ml_codex_v3.md` | Table 3 comparison gate | `pending` |
| `M2` | `stage_b` | `rm101_ml_openrouter_nemotron_v3` | `rm101_ml_openrouter_nemotron_v3` | RM101 | `openrouter` | `nvidia/nemotron-3-super-120b-a12b:free` | `rich` | `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py runtime.action=preflight +runs=rm101_ml_openrouter_nemotron_v3 && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python main.py +runs=rm101_ml_openrouter_nemotron_v3` | `artifacts/paper/rm101_ml_openrouter_nemotron_v3` | `doc/experiments/handoff/results/rm101_ml_openrouter_nemotron_v3.md` | Table 3 comparison gate | `pending` |
| `M1` | `stage_c` | `ottawa_ml` | `ottawa_ml_main_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/ottawa_ml_main_v1` | `artifacts/paper/ottawa_ml_main_v1` | `doc/experiments/handoff/results/ottawa_ml_main_v1.md (expected)` | Table 1 main result | `locked_by_selection` |
| `M2` | `stage_c` | `ottawa_torch` | `ottawa_torch_main_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/ottawa_torch_main_v1` | `artifacts/paper/ottawa_torch_main_v1` | `doc/experiments/handoff/results/ottawa_torch_main_v1.md (expected)` | Table 1 path comparison | `locked_by_selection` |
| `M1` | `stage_c` | `rm101_ml` | `rm101_ml_main_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/rm101_ml_main_v1` | `artifacts/paper/rm101_ml_main_v1` | `doc/experiments/handoff/results/rm101_ml_main_v1.md (expected)` | Table 1 main result | `locked_by_selection` |
| `M2` | `stage_c` | `rm101_torch` | `rm101_torch_main_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/rm101_torch_main_v1` | `artifacts/paper/rm101_torch_main_v1` | `doc/experiments/handoff/results/rm101_torch_main_v1.md (expected)` | Table 1 path comparison | `locked_by_selection` |
| `M2` | `stage_d` | `ottawa_ml` | `ottawa_ml_intermediate_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/ottawa_ml_intermediate_v1` | `artifacts/paper/ottawa_ml_intermediate_v1` | `doc/experiments/handoff/results/ottawa_ml_intermediate_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `rm101_ml` | `rm101_ml_intermediate_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/rm101_ml_intermediate_v1` | `artifacts/paper/rm101_ml_intermediate_v1` | `doc/experiments/handoff/results/rm101_ml_intermediate_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `ottawa_torch` | `ottawa_torch_module_runtime_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/ottawa_torch_module_runtime_v1` | `artifacts/paper/ottawa_torch_module_runtime_v1` | `doc/experiments/handoff/results/ottawa_torch_module_runtime_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `rm101_torch` | `rm101_torch_module_runtime_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/rm101_torch_module_runtime_v1` | `artifacts/paper/rm101_torch_module_runtime_v1` | `doc/experiments/handoff/results/rm101_torch_module_runtime_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `ottawa_torch` | `ottawa_torch_gated_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/ottawa_torch_gated_v1` | `artifacts/paper/ottawa_torch_gated_v1` | `doc/experiments/handoff/results/ottawa_torch_gated_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `rm101_torch` | `rm101_torch_gated_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/rm101_torch_gated_v1` | `artifacts/paper/rm101_torch_gated_v1` | `doc/experiments/handoff/results/rm101_torch_gated_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `ottawa_torch` | `ottawa_torch_attention_v1` | Ottawa | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/ottawa_torch_attention_v1` | `artifacts/paper/ottawa_torch_attention_v1` | `doc/experiments/handoff/results/ottawa_torch_attention_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |
| `M2` | `stage_d` | `rm101_torch` | `rm101_torch_attention_v1` | RM101 | `selected_global_best_backend.provider` | `selected_global_best_backend.model` | `rich` | `python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/rm101_torch_attention_v1` | `artifacts/paper/rm101_torch_attention_v1` | `doc/experiments/handoff/results/rm101_torch_attention_v1.md (expected)` | Table 2 ablation | `locked_by_selection` |

## Status Vocabulary

- `pass`: artifact bundle 与结果证据完整。
- `pass_with_local_incident`: 有历史 clean pass 证据，但当前本机最近一次复现出现 transport / environment incident。
- `fail`: 已有结果证据明确失败。
- `pending`: 预期要跑，但当前还没有足够证据写成 pass 或 fail。
- `locked_by_selection`: 必须等 `selected_global_best_backend.selected_from_stage_b=true` 才允许执行。
- `no_evidence`: 既无结果文件，也无完整 artifact bundle；当前总矩阵保留该词汇，但没有把它用于现有 rows。

## Proving Lane

Supervisor proving lane 只回答一个问题：

`raw plan -> StepPlan -> validated_dag.json -> compile_dag_for_path() -> ml artifacts`

固定规则：

- 这条 lane 只做 engineering qualification，不写 `01_result_ledger.md`
- 当前仓库里真实存在的 proving preset 只有：
  - `ottawa_ml_codex_proving`
  - `ottawa_ml_openrouter_glm_proving`
- 当前没有：
  - `rm101_ml_codex_proving`
  - `rm101_ml_openrouter_glm_proving`
- proving lane 的结果位置以 `artifact_dir` 为准；`result_md` 列固定写 `n/a (proving lane)`
- proving artifact contract 固定为：
  - `validated_dag.json`
  - `compiled_dag_manifest.json`
  - `feature_pipeline.json`
  - `feature_list.json`
  - `feature_separability_summary.json`
  - `artifact_index.json`
  - `metrics.json`
  - `final_report.md`
  - `step_plan.json`

当前 `ottawa_ml_openrouter_glm_proving` 的文档状态固定记为 `pass_with_local_incident`：仓库内有 clean pass artifact bundle，但最近一次本机复现暴露过 transport/proxy incident，因此不写成“完全稳定复现”。

## Simple Qualification Lane

`simple_fullchain` lane 只回答一个更现实的问题：

`plan -> execute -> reflect -> compile -> inquirer -> report`

能否在真实数据上稳定闭环，同时继续遵守：

- `validated DAG JSON -> compile_dag_for_path()`
- 同一套核心 artifact schema
- `workflow_state.json` 只保存状态快照与 `artifact_index_path`

固定规则：

- simple qualification lane 不写 `doc/experiments/01_result_ledger.md`
- simple qualification lane 不进入 `doc/experiments/02_main_tables.md`
- simple qualification lane 的成功只表示 runtime closure on real data，不表示 formal paper pass
- 当前 simple qualification round 只做：
  - `ottawa_ml_codex_simple`
  - `rm101_ml_codex_simple`
- 执行顺序固定为 Ottawa 先、RM101 后；如果 Ottawa 未 clean pass，就停止本轮，不继续 RM101 或 OpenRouter simple
- 当前已知结果：
  - `ottawa_ml_codex_simple = pass_with_local_incident`
  - `rm101_ml_codex_simple = fail`
- simple qualification artifact contract 固定为：
  - `validated_dag.json`
  - `compiled_dag_manifest.json`
  - `feature_pipeline.json`
  - `feature_list.json`
  - `feature_separability_summary.json`
  - `artifact_index.json`
  - `metrics.json`
  - `final_report.md`
  - `workflow_state.json`

## Canonical And Active Comparison

当前 formal paper presets 仍主要是：

- `ottawa_ml`
- `rm101_ml`
- `ottawa_torch`
- `rm101_torch`

当前 active Stage B comparison round 固定为：

- `codex_cli / gpt-5.3-codex`
- `openrouter / nvidia/nemotron-3-super-120b-a12b:free`

Stage B 只在 canonical `ml` mainline 上比较 backend，并且本轮统一使用 `v2` formal rows：

- `ottawa_ml_codex_v2`
- `ottawa_ml_openrouter_nemotron_v3`
- `rm101_ml_codex_v2`
- `rm101_ml_openrouter_nemotron_v3`

每条 comparison row 都必须满足：

- rich lane 正常完成
- DAG 深度 `3 <= depth <= 8`
- `validated_dag.json`、`compiled_dag_manifest.json`、`feature_pipeline.json`、`feature_list.json`、`feature_separability_summary.json`、`artifact_index.json`、`metrics.json`、`final_report.md` 齐全

历史 comparison evidence 继续保留在：

- `doc/experiments/01_result_ledger.md`
- `doc/experiments/incidents/03_openrouter_api_analysis.md`

但它们不再代表当前 active round，也不会进入 main tables。

## Selection Unlock Rule

`selected_global_best_backend` 仍以 `doc/experiments/01_result_ledger.md` 顶部 YAML block 为唯一事实源。

Stage C 与 Stage D 只有在以下条件成立后才解锁：

- `selected_global_best_backend.selected_from_stage_b=true`
- 对应 backend 在 Ottawa + RM101 的 Stage B rows 上同时满足：
  - `keep=accept`
  - `artifact_contract_pass=pass`
  - `feature_separability_pass=pass`
  - `selection_eligible=yes`

在此之前，所有 main / ablation rows 一律记为 `locked_by_selection`。

## Appendix: Stage A Pilot And Historical Notes

Stage A pilot 继续保留，但不进入首屏总矩阵。对应命令与 wrapper 仍可按需使用：

- `python main.py +runs=ottawa_ml_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/ottawa_ml_pilot_v1`
- `python main.py +runs=ottawa_torch_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/ottawa_torch_pilot_v1`
- `python main.py +runs=rm101_ml_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/rm101_ml_pilot_v1`
- `python main.py +runs=rm101_torch_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/rm101_torch_pilot_v1`

历史 comparison 失败 row 继续保留在 ledger / incidents，用于事故追踪，不进入总矩阵和 main tables：

- `ottawa_ml_codex_v1`
- `ottawa_ml_codex_v2`
- `ottawa_ml_openrouter_glm_v1`
- `ottawa_ml_openrouter_glm_v2`
- `ottawa_ml_openrouter_v1`
- `rm101_ml_codex_v1`
- `rm101_ml_codex_v2`
- `rm101_ml_openrouter_glm_v1`
- `rm101_ml_openrouter_glm_v2`
- `rm101_ml_openrouter_v1`
