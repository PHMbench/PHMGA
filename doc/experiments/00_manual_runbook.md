# Manual Experiment Runbook

`scripts/sh/` 只提供对本 runbook 常用命令的薄包装；正式、权威的实验命令与顺序仍以本文件为准。

默认执行语义只有两条：

- pilot = `offline_stub` smoke
- formal main = provider-backed research run，默认 `stepfun/step-3.5-flash:free`

前端 orchestration 已统一为 `PHMState + LangGraph StateGraph`；本 runbook 只关心正式入口 `main.py` 的实验执行，不再描述旧脚本循环。

## Pilot Baseline

- Pilot 默认基线：`llm.mode=offline_stub`
- `model.ml.output_policy=terminal_only`
- `model.torch.phase=compiled`
- `model.torch.output_policy=terminal_only`

## Formal Main Default

- formal main runs 统一使用 provider-backed LLM：
  - `llm.mode=provider`
  - `llm.provider=openrouter`
  - `llm.model=stepfun/step-3.5-flash:free`
- 如果手动做 provider ablation，建议显式写出同一组 provider 覆盖项，避免依赖默认值

## Stage A: Pilot Smoke

### Exit Criteria

- `preflight` 成功
- `run_case` 成功
- 输出目录中存在 graph-dependent artifacts 与 `final_report.md`
- pilot 结果登记进 `doc/experiments/01_result_ledger.md`

### Ottawa

```bash
python main.py runtime.action=preflight +runs=ottawa_ml_test
python main.py +runs=ottawa_ml_test runtime.output_dir=artifacts/paper/ottawa_ml_pilot_v1

python main.py runtime.action=preflight +runs=ottawa_torch_test
python main.py +runs=ottawa_torch_test runtime.output_dir=artifacts/paper/ottawa_torch_pilot_v1
```

### RM101

```bash
python main.py runtime.action=preflight +runs=rm101_ml_test
python main.py +runs=rm101_ml_test runtime.output_dir=artifacts/paper/rm101_ml_pilot_v1

python main.py runtime.action=preflight +runs=rm101_torch_test
python main.py +runs=rm101_torch_test runtime.output_dir=artifacts/paper/rm101_torch_pilot_v1
```

## Stage B: Formal Main Runs

### Entry Rule

- 只有当对应 dataset 的 pilot 已通过并已记账，才进入 formal run
- formal main 默认就是 `provider-backed LLM + terminal_only + compiled`
- 当前正式 main preset 已在 `config/runs/{ottawa,rm101}_{ml,torch}.yaml` 中显式写入：
  - `llm.mode=provider`
  - `llm.provider=openrouter`
  - `llm.model=stepfun/step-3.5-flash:free`

```bash
python main.py +runs=ottawa_ml runtime.output_dir=artifacts/paper/ottawa_ml_main_v1
python main.py +runs=ottawa_torch runtime.output_dir=artifacts/paper/ottawa_torch_main_v1

python main.py +runs=rm101_ml runtime.output_dir=artifacts/paper/rm101_ml_main_v1
python main.py +runs=rm101_torch runtime.output_dir=artifacts/paper/rm101_torch_main_v1
```

## Stage C: Method Ablations

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

## Stage D: Framework Ablation

```bash
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/ottawa_ml_openrouter_v1
python main.py +runs=rm101_ml llm.mode=provider llm.provider=openrouter llm.model=stepfun/step-3.5-flash:free runtime.output_dir=artifacts/paper/rm101_ml_openrouter_v1
```

## Recording Rule

- 每次运行后，必须把结果写入 `doc/experiments/01_result_ledger.md`
- 主结果表只从 ledger 抽取，不手填孤立数值
- 如果某次实验失败，也要登记为 `keep=reject`，并在 `note` 中写明失败原因
