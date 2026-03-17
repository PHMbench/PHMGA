# Manual Experiment Runbook

## Fixed Baselines

- `llm.mode=offline_stub`
- `model.ml.output_policy=terminal_only`
- `model.torch.phase=compiled`
- `model.torch.output_policy=terminal_only`

## Stage A: Pilot

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
- formal run 统一使用 `offline_stub + terminal_only + compiled`

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
python main.py +runs=ottawa_ml llm.mode=provider llm.provider=openrouter runtime.output_dir=artifacts/paper/ottawa_ml_openrouter_v1
python main.py +runs=rm101_ml llm.mode=provider llm.provider=openrouter runtime.output_dir=artifacts/paper/rm101_ml_openrouter_v1
```

## Recording Rule

- 每次运行后，必须把结果写入 `doc/experiments/01_result_ledger.md`
- 主结果表只从 ledger 抽取，不手填孤立数值
- 如果某次实验失败，也要登记为 `keep=reject`，并在 `note` 中写明失败原因
