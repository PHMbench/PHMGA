# 人工验收指南（PHMGA）：TSPN + LLM Provider（Gemini/GLM-4.7-Flash/DeepSeek）+ NVTA 工作流

本文件用于你在本机训练环境（推荐：`conda activate agent`）对本次合并内容做人工验收。

> 说明：当前容器环境可能未安装 `torch`，因此 TSPN 训练相关步骤请在本机/训练机执行。
>
> 说明：若你在容器里测试 GLM，可能会遇到 `open.bigmodel.cn` 无法解析（DNS / 网络策略限制）。此时请在你本机/训练机（可联网、可解析 DNS）执行第 7 节。

---

## TL;DR：最小可复现闭环（不含 TSPN 训练）

```bash
# 1) 自测
pytest -q

# 2) 运行 builder 生成 state（会输出 *.pkl 到 state_save_path）
python main.py case1 --config <CASE_YAML>

# 3) NVTA workflow：state -> datasets -> shallow ml -> report
python scripts/export_node_datasets.py --state <STATE_PKL> --out-dir generated_datasets --stage processed
python scripts/train_shallow_ml_from_npz.py --dataset-dir generated_datasets --out-md shallow_ml_results.md --out-pkl ml_results.pkl
python scripts/generate_report_from_state.py --state <STATE_PKL> --ml-results ml_results.pkl --out final_report.md
```

快捷运行（新增 scripts 封装）：

```bash
# 通用入口（默认 conda env: agent）
scripts/run_case.sh --config config/tspn_case_exp_ottawa.yaml --dry-run

# 预检查（provider/model、数据路径、算子/依赖）
python main.py preflight --config config/tspn_case_exp_ottawa.yaml

# Ottawa 常规 case
scripts/run_ottawa.sh --dry-run

# Ottawa TSPN 固定 ref/test ID case
scripts/run_tspn_ottawa.sh --dry-run

# 参数化切换（无痛换 profile / dataset）
scripts/run_case.sh --config config/tspn_case_exp_ottawa.yaml --profile tspn_basic --dataset Ottawa --dry-run
```

产物检查点（成功标准）：
- `final_report.md`
- `ml_results.pkl`
- `shallow_ml_results.md`
- `save/<case_name>/final_dag.png`（或你设置的 `PHM_SAVE_DIR` 目录下）

---

## 0. 准备

1) 进入仓库根目录：

```bash
cd /home/user/LQ/B_Signal/PHMGA
```

2) 准备 `.env`（LLM 配置）：

```bash
cp .env.example .env
```

3) 选择一个 provider（只需要一种）：

- Gemini：
  - `.env` 设置：`LLM_PROVIDER=gemini` + `GEMINI_API_KEY=...`
- GLM（OpenAI-compatible）：
  - `.env` 设置：`LLM_PROVIDER=glm` + `GLM_API_KEY=...` + `GLM_API_BASE=...`
- DeepSeek（OpenAI-compatible）：
  - `.env` 设置：`LLM_PROVIDER=deepseek` + `DEEPSEEK_API_KEY=...` + `DEEPSEEK_API_BASE=...`
- 任何 OpenAI-compatible 网关：
  - `.env` 设置：`LLM_PROVIDER=openai_compatible` + `OPENAI_API_KEY=...` + `OPENAI_BASE_URL=...`

4) 依赖矩阵（容器 vs 本机）

- 容器（可能无 `torch`）：
  - ✅ 可跑：`pytest -q`、`python scripts/* --help`、NVTA workflow 的浅层 ML 训练/报告
  - ❌ 不保证可跑：TSPN（需要 `torch`）
- 本机训练环境（推荐：`conda activate agent`）：
  - ✅ 可跑：包含 TSPN 的完整闭环（确保已安装 `torch`）

5) 本机环境的依赖检查（只在本机执行）：

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 0.1 pytest 扩展测试开关（可选）

默认 `pytest -q` 会跳过需要联网或需要外部数据/torch 的测试。你可以用以下环境变量打开：

```bash
# 1) 启用 torch 相关 smoke/契约测试（TSPN forward/train、DAG->TSPN init 等）
PHM_ENABLE_TORCH_TESTS=1 pytest -q

# 2) 启用 vibench data_factory 的端到端测试（需要本机可访问 vibench 代码路径与依赖）
PHM_ENABLE_VIBENCH_TESTS=1 pytest -q

# 3) 启用 GLM 在线联通测试（需要 GLM_API_KEY / GLM_API_BASE 且网络可用）
PHM_ENABLE_GLM_TESTS=1 pytest -q
```

---

## 1) 自测（你本机复现）

```bash
pytest -q
```

期望：
- `passed` 全部通过（容器里已通过，这里确保你本机环境一致）

---

## 2) Builder（构建 DAG）+ 保存 state

选择一个 case（示例 `config/case_exp_ottawa.yaml`），直接运行：

```bash
python main.py case1 --config config/case_exp_ottawa.yaml
```

期望：
- 生成 `state_save_path` 指向的 `*.pkl`
- 控制台显示 builder workflow 正常迭代结束（不会无限循环）

重跑提示（很常见）：
- 如果 `state_save_path` 已存在，runner 会跳过 builder（直接 load state）
- 需要重建时，删除旧 state 再跑：

```bash
rm -f <state_save_path>
python main.py case1 --config <CASE_YAML>
```

入口差异提示：
- 本仓库当前入口是：`python main.py case1 --config <CASE_YAML>`
- 不同于某些 PHM-Vibench 风格入口：`python main.py --config <yaml>`

---

## 3) NVTA workflow（导出数据集 → 浅层 ML → 报告）

假设上一步生成的 state 为：
- `<STATE_PKL>=/path/to/exp2.5built_state_ottawa.pkl`

前置检查点（否则导出会为空）：
- `export_node_datasets.py` 依赖 processed 节点的 `meta.saved.ref_path / meta.saved.tst_path`
- 若 `generated_datasets/` 导出为 0 个：
  - 回到 **Step 2** 重建 state（确保 builder/execute 产生 processed nodes 并保存 features）

### 3.1 导出节点数据集

```bash
python scripts/export_node_datasets.py --state <STATE_PKL> --out-dir generated_datasets --stage processed
```

期望：
- `generated_datasets/` 下出现若干 `*_dataset.npz`

> 若你需要在报告阶段计算 test 指标（不建议默认开启），追加：
> `--allow-test-labels-for-reporting`

### 3.2 训练浅层 ML（RandomForest 等）

```bash
python scripts/train_shallow_ml_from_npz.py --dataset-dir generated_datasets --out-md shallow_ml_results.md --out-pkl ml_results.pkl
```

期望：
- 生成 `shallow_ml_results.md`（markdown 表格）
- 生成 `ml_results.pkl`

### 3.3 从 state + ml_results 生成最终报告

```bash
python scripts/generate_report_from_state.py --state <STATE_PKL> --ml-results ml_results.pkl --out final_report.md
```

期望：
- 生成 `final_report.md`
- `save/<case_name>/final_dag.png`（或你设置的 `PHM_SAVE_DIR` 目录下）

---

## 4) Executor graph（inquire → prepare → train → report）

在对应 `config/case*.yaml` 里打开执行：

- 新增/修改：
  - `run_executor: true`
  - `train_backend: shallow`（只跑浅层 ML）或 `both`（浅层+TSPN）
  - （可选）`allow_test_labels_for_reporting: true`

最小 YAML 片段示例（只写 key，不写你的绝对路径）：

```yaml
run_executor: true
train_backend: shallow   # shallow | tspn | both
model_config_path: config/model_tspn_basic.yaml  # 当 train_backend=tspn|both 时需要
allow_test_labels_for_reporting: false           # 默认 false；仅报告阶段可选 true
```

然后运行：

```bash
python main.py case1 --config config/case_exp_ottawa.yaml
```

期望：
- executor workflow 执行并生成 report（`report_path`）

---

## 5) TSPN 训练（需要 torch）

### 5.1 配置

使用 `config/model_tspn_basic.yaml` 作为起点，并保证：
- `model.in_dim == L`（信号长度）
- `model.in_channels == C`（`dag_state.channels` 的通道数）
- `model.num_classes` 与数据集标签类别数一致

在 case yaml 增加：
- `run_executor: true`
- `train_backend: tspn`（或 `both`）
- `model_config_path: config/model_tspn_basic.yaml`

### 5.2 运行

```bash
python main.py case1 --config config/<your_case>.yaml
```

期望（在 `save/<case>/<timestamp>/`）：
- `metrics.json`
- `predictions.csv`
- `explain/operator_importance.json`
- `explain/wavefilters_params.json`（若启用 WF）

---

## 6) 常见问题（快速定位）

1) `ModuleNotFoundError: No module named 'torch'`
   - 说明当前环境未安装 PyTorch；请切到你的训练环境（`conda activate agent`）并安装 `torch`
2) GLM/DeepSeek 连接失败
   - 检查 `.env`：`*_API_KEY`、`*_BASE_URL` 是否正确
   - 检查是否已安装 `langchain_openai`（OpenAI-compatible 必需）
3) `in_dim/in_channels mismatch`
   - 说明 `config/model_tspn_basic.yaml` 与数据不一致，按上面 5.1 修正

---

## 7) GLM-4.7-Flash 在线联通测试（`conda activate agent`）

> 目标：验证你的 GLM API Key/Base/模型名配置正确，并验证 PHMGA 的 `get_llm()` 能走 OpenAI-compatible 路径。

### 7.1 一键 smoke test（推荐）

```bash
conda activate agent
python scripts/llm_smoke_test_glm.py
```

如果你更习惯用 `conda run`（不进入交互式 shell）：

```bash
conda run -n agent python scripts/llm_smoke_test_glm.py
```

成功标准：
- `HTTP_STATUS=200`，响应里能看到 assistant 内容（接近 `OK`）
- `PHMGA_LLM_CLASS=ChatOpenAI`，并输出 `PHMGA_LLM_REPLY=...OK...`

说明：
- 该脚本会在运行时强制 `LLM_PROVIDER=glm`（避免你终端里已有 `LLM_PROVIDER=gemini` 导致误判）。

### 7.2 若失败的典型原因

- `NameResolutionError / DNS`：当前网络无法解析 `open.bigmodel.cn`（需要换可联网/可解析 DNS 的网络环境）
- `401/403`：`GLM_API_KEY` 错误或没有权限
- `404`：`GLM_API_BASE` 可能写错（不要写到 `/chat/completions` 结尾）

4) `export_node_datasets.py` 导出为空
   - 多数情况是 state 里没有包含带 `meta.saved` 的 processed nodes
   - 回到 Step 2 重建 state，或检查是否确实执行了 execute（产生并保存节点产物）
