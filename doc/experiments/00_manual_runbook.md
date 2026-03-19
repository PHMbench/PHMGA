# Manual Experiment Runbook

`scripts/sh/` 只提供对本 runbook 常用命令的薄包装；正式、权威的实验命令与顺序仍以本文件为准。

如果要把实验交给其他 Codex CLI worker 执行，统一使用：

- `doc/experiments/04_codex_cli_handoff.md`
- `doc/experiments/05_worker_result_template.md`
- `doc/experiments/06_multi_agent_merge_checklist.md`
- `doc/experiments/handoff/*.md`

默认执行语义只有两条：

- pilot = `offline_stub` smoke
- formal main = provider-backed research run，当前已冻结为 `codex_cli + gpt-5.3-codex`
- root `config/config.yaml` 继续只是 smoke/development baseline；不要把它当成论文主实验默认

前端 orchestration 已统一为 `PHMState + LangGraph StateGraph`；本 runbook 只关心正式入口 `main.py` 的实验执行，不再描述旧脚本循环。

## Research Closure Milestones

本 runbook 的 Stage A/B/C/D 是实验执行顺序，不是研究主线定义。  
研究收口顺序固定为：

1. `M0: Agent Core`
   - 先证明 `plan -> execute -> dag_quality -> reflect -> finish|rollback` 能稳定导出 compileable 的 `validated DAG JSON`。
2. `M1: Dataset-Level Evidence`
   - 再证明同一个 DAG 在真实 `train/val/test` 上具有 split-level sampled dataset evidence。
3. `M2: Comparison Layer`
   - 最后才比较 path、provider candidate 与 runtime 设定。

其中 canonical diagnosis backend 先固定为 `ml`；`torch` 当前继续作为比较层最小实现。

## Pilot Baseline

- Pilot 默认基线：`llm.mode=offline_stub`
- `model.ml.output_policy=terminal_only`
- `model.torch.phase=compiled`
- `model.torch.output_policy=terminal_only`

## Formal Main Rule

- formal main runs 统一使用 provider-backed LLM：
  - `llm.mode=provider`
  - `llm.provider=codex_cli`
  - `llm.model=gpt-5.3-codex`
- `stepfun/step-3.5-flash:free` 继续保留为 qualification candidate，不再承担 formal main 默认
- 当前 `config/runs/{ottawa,rm101}_{ml,torch}.yaml` 已重新冻结到 Codex tuple；formal main 与 method ablations 默认直接继承 preset
- `scripts/sh/main/*.sh` 支持：
  - `PHMGA_LLM_PROVIDER`
  - `PHMGA_LLM_MODEL`
  - `PHMGA_OUTPUT_DIR`
  - 但正式 evidence run 默认不应覆盖 frozen tuple

## Stage A: Pilot Smoke

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

## Stage B: Provider Qualification

### Entry Rule

- 只有当对应 dataset 的 pilot 已通过并已记账，才进入 formal run
- qualification 只验证 provider/model tuple 的可达性、structured compatibility、retry-once 稳定性和 artifact 完整性
- 当前保留两类 Stage B 运行：
  - `openrouter + stepfun/step-3.5-flash:free` 作为 candidate qualification
  - `codex_cli + gpt-5.3-codex` 作为 frozen formal-main backend sanity check
- qualification 先固定在 `ml` path 上做，避免把 runtime 变量和 provider 变量混在一起

```bash
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/ottawa_ml_openrouter_v1
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=codex_cli llm.model=gpt-5.3-codex runtime.output_dir=artifacts/paper/ottawa_ml_codex_v1

python main.py +runs=rm101_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/rm101_ml_openrouter_v1
python main.py +runs=rm101_ml llm.mode=provider llm.provider=codex_cli llm.model=gpt-5.3-codex runtime.output_dir=artifacts/paper/rm101_ml_codex_v1
```

对应 shell wrapper：

```bash
./scripts/sh/ablation/provider/ottawa_ml_openrouter.sh
./scripts/sh/ablation/provider/ottawa_ml_codex.sh
./scripts/sh/ablation/provider/rm101_ml_openrouter.sh
./scripts/sh/ablation/provider/rm101_ml_codex.sh
```

## Stage C: Formal Main Runs

### Entry Rule

- Formal Main 现在固定使用 Codex tuple
- `scripts/sh/main/*.sh` 默认即可直接运行；只有做诊断时才应覆盖 provider/model
- 当前 formal main 的主线证明面优先看 `ml`；`torch` 结果记录为 path comparison，不重新定义 PHMGA 核心

正式命令模板：

```bash
python main.py +runs=ottawa_ml runtime.output_dir=artifacts/paper/ottawa_ml_main_v1
python main.py +runs=ottawa_torch runtime.output_dir=artifacts/paper/ottawa_torch_main_v1

python main.py +runs=rm101_ml runtime.output_dir=artifacts/paper/rm101_ml_main_v1
python main.py +runs=rm101_torch runtime.output_dir=artifacts/paper/rm101_torch_main_v1
```

## Stage D: Method Ablations

Stage D 统一视为 comparison layer；这些实验不反向定义 PHMGA 主线。

### Output Policy

```bash
python main.py +runs=ottawa_ml model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/ottawa_ml_intermediate_v1
python main.py +runs=rm101_ml model.ml.output_policy=include_intermediate_features runtime.output_dir=artifacts/paper/rm101_ml_intermediate_v1
```

### GraphModule / Learnable Control

```bash
python main.py +runs=ottawa_torch model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/ottawa_torch_module_runtime_v1
python main.py +runs=rm101_torch model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/rm101_torch_module_runtime_v1

python main.py +runs=ottawa_torch model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/ottawa_torch_gated_v1
python main.py +runs=rm101_torch model.torch.phase=learnable_control model.torch.control.default_mode=gated runtime.output_dir=artifacts/paper/rm101_torch_gated_v1

python main.py +runs=ottawa_torch model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/ottawa_torch_attention_v1
python main.py +runs=rm101_torch model.torch.phase=learnable_control model.torch.control.default_mode=attention model.torch.control.attention_heads=2 runtime.output_dir=artifacts/paper/rm101_torch_attention_v1
```

### WaveFilters Family

当前 planner 尚未主动生成这组算子，因此建议把这组实验当成 runtime-enhanced transform track 单独记账。

推荐顺序：

```bash
# 先用 fixed/module runtime 验证 GraphModule 稳定
python main.py +runs=ottawa_torch model.torch.phase=module_runtime model.torch.module_runtime.enabled=true runtime.output_dir=artifacts/paper/ottawa_torch_wavefilters_baseline_v1

# WaveFilters family 的正式对照在记录上固定为：
# no-wavefilter -> wavefilters -> ricker -> morlet -> other family
```

## Recording Rule

- 如果交给其他 Codex CLI worker：
  - 先写 `doc/experiments/handoff/results/<experiment_id>.md`
  - 再写 `doc/experiments/01_result_ledger.md`
- 每次运行后，必须把结果写入 `doc/experiments/01_result_ledger.md`
- 主结果表只从 ledger 抽取，不手填孤立数值
- 如果某次实验失败，也要登记为 `keep=reject`，并在 `note` 中写明失败原因
