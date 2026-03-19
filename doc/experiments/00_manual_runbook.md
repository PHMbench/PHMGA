# Manual Experiment Runbook

`scripts/sh/` 只提供对本 runbook 常用命令的薄包装；正式、权威的实验命令、阶段定义、backend 选择规则和结果回写顺序仍以本文件为准。

如果要把实验交给其他 Codex CLI worker 执行，统一使用：

- `doc/experiments/04_codex_cli_handoff.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`
- `doc/experiments/handoff/*.md`

## Default Facts

- 两个真实数据集固定为：
  - `Ottawa`
  - `RM101`
- root `config/config.yaml` 继续只是 smoke/development baseline；不要把它当成论文主实验默认
- `ml` 是 canonical diagnosis mainline
- `torch` 只做 path comparison
- “多个后端”只指 **LLM/provider backend**，不与 `dag_only / ml / torch` 混写
- 外层执行控制面仍是 `Codex CLI worker`
- 但 `worker tool != experiment backend`：
  - worker 使用 Codex CLI 接任务
  - 实验内部实际使用的 backend tuple 由 Stage B 和 ledger 顶部的 `selected_global_best_backend` 决定

## Research Closure Milestones

本 runbook 的 Stage A/B/C/D 是实验执行顺序，不是研究主线定义。研究收口顺序固定为：

1. `M0: Agent Core`
   - 先证明 `plan -> execute -> dag_quality -> reflect -> finish|rollback` 能稳定导出 compileable 的 `validated DAG JSON`。
2. `M1: Dataset-Level Evidence`
   - 再证明同一个 DAG 在真实 `train/val/test` 上具有 split-level sampled dataset evidence。
3. `M2: Comparison Layer`
   - 最后才比较 path、backend candidate 与 runtime 设定。

其中 canonical diagnosis backend 先固定为 `ml`；`torch` 当前继续作为比较层最小实现。

## Stage A: Pilot Smoke

### Fixed Settings

- `llm.mode=offline_stub`
- `model.ml.output_policy=terminal_only`
- `model.torch.phase=compiled`
- `model.torch.output_policy=terminal_only`

### Exit Criteria

- `preflight` 成功
- `run_case` 成功
- 输出目录中存在 graph-dependent artifacts 与 `final_report.md`
- pilot 结果登记进 `doc/experiments/01_result_ledger.md`

### Ottawa

```bash
python main.py runtime.action=preflight +runs=ottawa_ml_test llm.mode=offline_stub
python main.py +runs=ottawa_ml_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/ottawa_ml_pilot_v1

python main.py runtime.action=preflight +runs=ottawa_torch_test llm.mode=offline_stub
python main.py +runs=ottawa_torch_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/ottawa_torch_pilot_v1
```

### RM101

```bash
python main.py runtime.action=preflight +runs=rm101_ml_test llm.mode=offline_stub
python main.py +runs=rm101_ml_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/rm101_ml_pilot_v1

python main.py runtime.action=preflight +runs=rm101_torch_test llm.mode=offline_stub
python main.py +runs=rm101_torch_test llm.mode=offline_stub runtime.output_dir=artifacts/paper/rm101_torch_pilot_v1
```

对应 shell wrapper：

```bash
./scripts/sh/pilot/01_ottawa_ml_pilot.sh
./scripts/sh/pilot/02_ottawa_torch_pilot.sh
./scripts/sh/pilot/03_rm101_ml_pilot.sh
./scripts/sh/pilot/04_rm101_torch_pilot.sh
```

## Stage B: Backend Comparison On Canonical ML Mainline

### Purpose

Stage B 不是单纯的 backend 打分。它先验证：

- PHMGA 全链路是否 work
- `validated DAG JSON -> compile_dag_for_path()` 是否成功
- `ml` path 是否真的产出可分离特征

只有在此基础上，backend 才有资格参与 `selected_global_best_backend` 选择。

### Candidate Registry

#### Codex candidate registry

- `codex_cli / gpt-5.4`
- `codex_cli / gpt-5.4-mini`
- `codex_cli / gpt-5.2`
- `codex_cli / gpt-5.3-codex`

#### OpenRouter candidate registry

- `openrouter / stepfun/step-3.5-flash:free`
- `openrouter / google/gemini-2.0-flash-exp`
- `openrouter / google/gemini-2.5-pro`
- `openrouter / openrouter/free`

约束写死：

- 每轮 Stage B 只允许激活 **1 个 codex tuple + 1 个 OpenRouter tuple**
- `openrouter/free` 只允许用于 `qualification / pre-screen`，不允许参与最终 `selected_global_best_backend` 选择
- 当前 active Stage B set 以 `doc/experiments/01_result_ledger.md` 顶部的 YAML block 为准
- 当前 shell wrapper 默认对齐的 active set 是：
  - `codex_cli / gpt-5.3-codex`
  - `openrouter / stepfun/step-3.5-flash:free`

### Entry Rule

- 对应 dataset 的 pilot 已完成并记账
- Stage B 固定只在 `ml` path 上跑，避免把 runtime 变量和 backend 变量混在一起
- worker 只允许运行当前 active set 对应的 comparison row

### Artifact Contract Gate

只有满足以下硬门槛，Stage B row 才能记 `artifact_contract_pass=pass`：

- `validated_dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `feature_list.json`
- `feature_separability_summary.json`
- `artifact_index.json`
- `metrics.json`
- `final_report.md`

### Feature Separability Gate

只有满足以下最小诊断证据，Stage B row 才能记 `feature_separability_pass=pass`：

- feature pipeline 非空
- 至少存在一份 feature-level evidence 证明特征没有全塌缩
- 至少有最小 separability summary

### Comparison Commands

```bash
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=codex_cli llm.model=gpt-5.3-codex runtime.output_dir=artifacts/paper/ottawa_ml_codex_v1
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/ottawa_ml_openrouter_v1

python main.py +runs=rm101_ml llm.mode=provider llm.provider=codex_cli llm.model=gpt-5.3-codex runtime.output_dir=artifacts/paper/rm101_ml_codex_v1
python main.py +runs=rm101_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/rm101_ml_openrouter_v1
```

对应 shell wrapper：

```bash
./scripts/sh/ablation/provider/ottawa_ml_codex.sh
./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh
./scripts/sh/ablation/provider/rm101_ml_codex.sh
./scripts/sh/ablation/provider/rm101_ml_openrouter.sh
```

### Backend Selection Rule

Stage B row 只有在同时满足以下条件后，才有资格参与 `selected_global_best_backend` 选择：

- `keep=accept`
- `artifact_contract_pass=pass`
- `feature_separability_pass=pass`

选择规则固定为：

1. 只比较在两个数据集上都通过 gate 的 backend
2. 主排序键：两个 `ml` comparison row 的 `macro_f1` 均值
3. 第一 tie-break：`accuracy` 均值
4. 第二 tie-break：incident/failure 更少者优先
5. 若仍持平，默认优先当前 active Codex tuple

如果只有一个 backend 同时通过两个数据集的 gate，则它自动成为 `selected_global_best_backend`。  
如果两个 backend 都未满足资格，则 `selected_global_best_backend` 保持 `pending`，Stage C 与 Stage D 不解锁。

## Stage C: Formal Main Using Selected Global-Best Backend

### Entry Rule

- `selected_global_best_backend.selected_from_stage_b=true`
- `selected_global_best_backend` 以 `doc/experiments/01_result_ledger.md` 顶部的 YAML block 为唯一事实源
- Stage C 的 4 条运行全部继承 selected backend

当前 shell wrapper 默认仍继承 `codex_cli / gpt-5.3-codex`。  
如果 selected backend 与 wrapper 默认值不同，worker 必须通过环境变量显式覆盖：

```bash
PHMGA_LLM_PROVIDER=<selected_provider> PHMGA_LLM_MODEL=<selected_model> ./scripts/sh/main/01_ottawa_ml_main.sh
```

### Formal Main Commands

```bash
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/ottawa_ml_main_v1
python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/ottawa_torch_main_v1

python main.py +runs=rm101_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/rm101_ml_main_v1
python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> runtime.output_dir=artifacts/paper/rm101_torch_main_v1
```

语义写死：

- `ml` 两条是 canonical diagnosis mainline
- `torch` 两条是同一 selected backend 下的 path comparison

## Stage D: Best-Backend Ablations

Stage D 统一视为 comparison layer；这些实验不反向定义 PHMGA 主线。  
当前 8 条 ablation 全部继承 `selected_global_best_backend`，不再按 backend 复制矩阵。

如果 selected backend 与当前 wrapper 默认值不同，同样必须通过 `PHMGA_LLM_PROVIDER` / `PHMGA_LLM_MODEL` 覆盖。

### Output Policy

```bash
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/ottawa_ml_intermediate_v1
python main.py +runs=rm101_ml llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/rm101_ml_intermediate_v1
```

### GraphModule / Learnable Control

```bash
python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/ottawa_torch_module_runtime_v1
python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/rm101_torch_module_runtime_v1

python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/ottawa_torch_gated_v1
python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/rm101_torch_gated_v1

python main.py +runs=ottawa_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/ottawa_torch_attention_v1
python main.py +runs=rm101_torch llm.mode=provider llm.provider=<selected_provider> llm.model=<selected_model> model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/rm101_torch_attention_v1
```

## Recording Rule

- 每次运行后，必须先写 `doc/experiments/handoff/results/<experiment_id>.md`
- 再写 `doc/experiments/01_result_ledger.md`
- coordinator 最后才允许更新 `doc/experiments/02_main_tables.md`

### Required Evidence Layers

#### 硬门槛 artifacts

- `validated_dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `feature_list.json`
- `feature_separability_summary.json`
- `artifact_index.json`
- `metrics.json`
- `final_report.md`

#### required evidence

- `progress record`

短期过渡规则：

- `progress_record` 仍允许由 `results/<experiment_id>.md` 提供 provisional evidence
- 但只要缺任一硬门槛 artifact，就不得 `accept`
- 若缺 `progress record`，最多记 `needs_rerun`，不得进入主表
