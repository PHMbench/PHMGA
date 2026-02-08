# Runbook：如何在 `conda activate agent` 下验证 LLM→DAG→TSPN

## 1) 环境
```bash
conda activate agent
cd /home/user/LQ/B_Signal/PHMGA
```

> 如果你只想离线跑通，不需要联网：设置 `FAKE_LLM=true` + `PHM_REPORT_MODE=template`。

---

## 2) pytest（推荐：先跑默认，再按需打开扩展）
### 2.1 默认（不强制联网/外部数据）
```bash
pytest -q
```

### 2.2 打开 torch/TSPN 相关测试
```bash
PHM_ENABLE_TORCH_TESTS=1 pytest -q \
  tests/test_explainable_ops_contract.py \
  tests/test_tspn_forward_and_train_smoke.py \
  tests/test_dag2tspn_init_from_filter.py
```

### 2.3 打开 vibench 端到端（Dummy_Data）
```bash
PHM_ENABLE_TORCH_TESTS=1 PHM_ENABLE_VIBENCH_TESTS=1 pytest -q \
  tests/test_full_agent_flow_vibench_tspn_report.py
```

### 2.4 打开 GLM 在线通讯（需要 `.env`/环境变量配置）
```bash
PHM_ENABLE_GLM_TESTS=1 pytest -q tests/test_glm_online_smoke.py
```

---

## 3) GLM 一键 smoke（不走 pytest）
```bash
python scripts/llm_smoke_test_glm.py
```

成功标准：
- `HTTP_STATUS=200`
- `PHMGA_LLM_CLASS=ChatOpenAI`
- `PHMGA_LLM_REPLY=...OK...`

---

## 4) 运行 case（真实 workflow）
离线建议：
- `.env` 中可不配 key
- 运行前：`export FAKE_LLM=true`、`export PHM_REPORT_MODE=template`

```bash
export FAKE_LLM=true
export PHM_REPORT_MODE=template
python main.py case1 --config <your_case_yaml>
```

vibench case 的关键字段（示例片段）：
```yaml
run_executor: true
train_backend: tspn
data:
  backend: vibench
  vibench_code_root: /home/user/LQ/B_Signal/vibench_fix/PHM-Vibench copy 2
  data_dir: /home/user/data/PHMbenchdata/PHM-Vibench
  metadata_file: metadata.xlsx
  dataset_name: Dummy_Data
  window_size: 4096
  stride: 512
  num_window: 8
  fs_hz: 12000
  debug: true
  debug_epochs: 1
```

