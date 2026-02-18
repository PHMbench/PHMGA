# RM101 上线验收证据（2026-02-17）

## 1) 预检基线

命令：

```bash
conda run -n agent python main.py preflight --config config/case_exp_gearbox_rm101.yaml
```

结果：
- `OK: True`
- Warnings:
  - `Unregistered operators: spectral_entropy, stft`
  - `Optional dependency 'nolds' is missing`
  - `Optional dependency 'graphviz' python package is missing`

## 2) 离线可复现跑通（FAKE_LLM + template report）

命令：

```bash
FAKE_LLM=true PHM_REPORT_MODE=template conda run -n agent python main.py case1 --config config/case_exp_gearbox_rm101.yaml
```

执行记录：
- 首次运行命中历史状态文件校验失败：缺少 `built_state.pkl.sha256`。
- 按失败分流处理：删除旧 `save/exp_gearbox_rm101/built_state.pkl` 后重跑成功。

重跑后关键输出：
- `Val acc: 0.2708333333333333`
- `Val macro_f1: 0.17124576510425438`
- `Artifacts: /home/user/LQ/B_Signal/PHMGA/save/exp_gearbox_rm101/20260217-112000`
- 报告保存：`save/exp_gearbox_rm101/final_report.md`

## 3) 验收文件检查（最小必过）

运行目录：`save/exp_gearbox_rm101/20260217-112000`

必备文件全部存在：
- `metrics.json`
- `dataset_manifest.json`
- `config_resolve.json`
- `model_config.resolved.yaml`
- `preflight_report.json`
- `predictions.csv`
- `explain/operator_importance.json`
- `save/exp_gearbox_rm101/final_report.md`

语义检查：
- `metrics.json` 含 `val_acc`, `val_macro_f1`
- `dataset_manifest.json` 显示：
  - `source_mode = vibench`
  - `dataset_name = RM_101_THU_GEARBOX`
- `config_resolve.json` 显示：
  - `autofit_dims = true`
  - `autofit_num_classes = true`
- 日志确认 executor 路径：
  - `save/exp_gearbox_rm101/run-1771298258/logs/events.jsonl`
  - 包含 `executor.path=tspn_fast_path`，且 `train/report` 节点执行成功

## 4) 可选在线联通验收

命令：

```bash
conda run -n agent python tests/llm_smoke_test_glm.py
```

结果（当前环境）：
- provider/model 解析正常（`LLM_PROVIDER=glm`, `MODEL=GLM-4.7-Flash`）
- 失败原因：网络 DNS 解析失败（`open.bigmodel.cn` 无法解析）
- 结论：在线链路未通过，但不影响离线验收闭环

## 5) 失败分流执行记录

已验证分流：
1. **状态校验失败**（`sha256` sidecar 缺失）  
   - 行动：清理旧 state 并重建；
   - 结果：Builder + Executor + Report 全流程通过。
2. **PNG 导出失败**（缺少 `graphviz`）  
   - 结果：自动降级为 `.dot`，非阻塞主链路。
