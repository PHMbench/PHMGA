# Agent 启动与闭环运行步骤（2.5）：DAG 初始化 → TSPN 配置 → 微调训练

本文件给出“你本机训练环境”中，从 0 启动 agent 的可执行步骤（命令级），并明确哪些步骤是 **当前已实现**、哪些是 **2.5 计划要实现**。

> 说明：容器里可能未安装 `torch`；TSPN 训练请在你的训练环境（例如 `conda activate LQ_signal` 或你自己的环境）执行。

---

## A. 当前已实现：Builder DAG + Executor（shallow / tspn / both）

### A1) 准备 `.env`（LLM provider）

仓库根目录：

```bash
cd /home/user/LQ/B_Signal/PHMGA
cp .env.example .env
```

最小配置（GLM 示例）：

```ini
LLM_PROVIDER=glm
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4
GLM_API_KEY=...
QUERY_GENERATOR_MODEL=GLM-4.7-Flash
```

> 验收：`python scripts/llm_smoke_test_glm.py` 返回 `HTTP_STATUS=200` 且 `PHMGA_LLM_REPLY=OK`。

### A2) 准备 case.yaml（以 `config/case_exp_ottawa.yaml` 为例）

关键字段（你需要按机器路径修改）：
- `save_dir/state_save_path/report_path`
- `metadata_path/h5_path`
- `ref_ids/test_ids`
- `run_executor: true`
- `train_backend: tspn | shallow | both`
- `model_config_path: config/model_tspn_basic.yaml`（当 `train_backend=tspn|both`）

### A3) 启动

```bash
python main.py case1 --config config/case_exp_ottawa.yaml
```

运行逻辑：
- 若 `state_save_path` 存在：跳过 Builder，直接 load
- 否则：执行 Builder（Plan→Execute→Reflect 循环）构建 DAG，并保存 state
- 若 `run_executor: true`：执行 Executor（inquire→prepare→train→report）

验收（最小）：
- state 保存成功（`*.pkl`）
- 若启用 executor：生成 `report_path`，并在 `save/<case>/<timestamp>/` 产生训练/解释性 artifacts

补充（可选）：Torch-side 算子契约测试（在安装了可用 torch 的环境）

```bash
PHM_ENABLE_TORCH_TESTS=1 pytest -q tests/test_explainable_ops_contract.py
PHM_ENABLE_TORCH_TESTS=1 pytest -q tests/test_explainable_feature_ops_contract.py
```

补充（可选）：Real data 快速 smoke（不依赖完整 builder）

```bash
# 以 Dummy_Data / 或你指定 dataset_name 的 vibench 配置为例
python main.py case1 --config <YOUR_CASE_YAML>
```

期望：
- 训练阶段首批数据进入 forward 时，能看到 `(B,L,C)` 的契约检查日志（若开启 debug）
- 输出 `TrainReport`（结构与 `doc/plan/2_5/AGENT_IO.md` 一致）
- 最终 `report_path` 中包含 TSPN 的指标与解释性摘要（operator importance / wavefilters 等）

---

## B. 2.5 目标：DAG 初始化 → 自动生成 TSPN 配置 → warm-start 微调（需要新增 agent）

你要的关键点是：“**让 agent 从数据初始化 DAG，然后把 DAG 的结构赋值给 TSPN（等价于生成/修改 `model_config.yaml`），再做微调训练**”。

为了工程可控与可复现，本项目采用 **Immutable Config**：
- Agent **只写** `model_config.yaml`（或输出 `ConfigPatch` 再由 runner 应用）
- 训练器读取 YAML 构建新模型（可选择 warm-start）
- 不在内存里热更新 `nn.Module` 结构（避免 optimizer/state_dict 复杂性）

### B1) DAG 初始化（数据驱动、确定性）

输入：data_factory 读入的原始多通道信号，输出：仅含 `ch1..chC` 的 roots DAG。

要求：
- roots 的 `results['ref'/'tst']` 是 `{sample_id: (1,L,1)}`
- roots 的 `meta` 至少包含：`fs/labels_ref/labels_tst`
- 推断：
  - `L -> model.in_dim`
  - `C -> model.in_channels`
  - `|unique(labels_ref)| -> model.num_classes`

### B2) DAG → TSPN 结构映射（BootstrapConfig）

建议的 MVP mapping（确定性）：
1) 只映射你 Unified 版本里最核心的 SP tokens：`I/WF/HT/FFT`
2) 定义 `dag_op_name -> tspn_token` 表（例）：
   - `identity -> I`
   - `hilbert* -> HT`
   - `fft* -> FFT`
   - `wavefilter* -> WF`
3) 用 DAG 深度决定 TSPN 的 layer：
   - 深度 d（从 root=0）出现的 token，构成 TSPN 的第 d+1 层并行 ops
   - 重复 token 用 occurrence_idx 编号（形成 `op_uid=L{layer}:{token}:{occ}`）
4) 对无法映射的 DAG op：
   - 保留在 DAG 作为非可微预处理，或丢弃但记录到 `meta.unmapped_ops`

输出：初始 `model_config.yaml`（或 `ConfigPatch`），满足 schema（`src/model/explainable/config_schema.py`）。

> 当前已实现：
> - `src/agents/dag_init_agent.py`：LLM 初始化 processed DAG（Executor graph 节点：`init_dag`）
> - `src/agents/tspn_bootstrap_agent.py`：确定性 DAG→TSPN 配置生成

### B3) Smoke Run（强制）

每次生成/修改 `model_config.yaml` 后必须先跑 smoke：
- `epochs=1`
- `batch_size=2`
- `subset=10 samples`

Smoke 失败返回：
- `INVALID_SHAPE`
- `INVALID_DIVISIBILITY`
- `COMPLEX_DTYPE`
并触发外环回退/重生成配置。

### B4) 微调训练（warm-start）

策略（建议写死在 train config 里）：
- Patch 很小（例如只禁用 1 个 op / 调 gate_temperature）：warm-start 上次 `checkpoint_best.pt`
- Patch 很大（增删层、改通道宽度）：从头训练（或部分加载，`strict=false`）

### B5) Explain → Reflect → ConfigPatch（闭环）

训练结束输出 `TrainReport`（见 `doc/plan/2_5/AGENT_IO.md`）：
- metrics + confusion matrix
- explain_summary（operator importance / WF bands / feature stats）
- reason_codes（Reflect 产出）

外环基于 reason_codes + explain_summary 输出下一轮 `ConfigPatch`：
- “删”= `disabled_ops[op_uid]=1e-6`
- “增/改”= 调整 layer token 列表、WF init、features、训练超参

---

## C. 最小实现建议（你下一步要让代码做什么）

为了尽快落地“DAG→TSPN→微调”，建议新增 2 个 agent：

1) `tspn_bootstrap_agent`（**确定性**，不需要 LLM）
   - 输入：`PHMState`（含 channels/L/labels_ref + DAG 摘要）
   - 输出：一个合法的 `model_config.yaml`（或 `ConfigPatch`）
2) `model_config_agent`（LLM）
   - 输入：上次 `model_config.yaml` + `TrainReport`
   - 输出：严格白名单的 `ConfigPatch`

验收标准：
- 同一份 state（相同 data + DAG）bootstrap 出的 YAML 必须完全一致
- patch 必须可被 schema 校验并应用
- warm-start 在小 patch 上能复用 checkpoint（可观察更快收敛）
