## PLAN（阶段任务 / 验收标准）—— 把 Unified-X 可解释网络集成到 PHMGA 并闭环自动配置

本 PLAN 只写“要做什么 + 如何验收”，不可变约定见 `SPEC.md`，I/O schema 见 `AGENT_IO.md`。

---

### 0) Agent 启动步骤（现有可跑链路 + 2.5 目标链路）

#### 0.1 现有可跑链路（PHMGA 当前实现）

目的：先把“数据初始化 DAG →（可选）executor 训练/报告”跑通，确保工程基座可复现。

1) 配置 LLM（只要能让 Builder/Reporter 调用即可，推荐 GLM）
   - 根目录准备 `.env`（参考 `.env.example`）
   - 最小字段示例：
     - `LLM_PROVIDER=glm`
     - `GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4`
     - `GLM_API_KEY=...`
     - `QUERY_GENERATOR_MODEL=GLM-4.7-Flash`
2) 准备 case.yaml（当前仓库使用 `config/case_*.yaml` 风格）
   - 关键字段：
     - `metadata_path / h5_path / ref_ids / test_ids`
     - `builder.min_depth / builder.max_depth`
     - `run_executor: true|false`
     - `train_backend: shallow|tspn|both`
     - `model_config_path: config/model_tspn_basic.yaml`（当 `train_backend=tspn|both`）
3) 启动
   - 只跑 Builder（构建 DAG）：`python main.py case1 --config config/case_exp_ottawa.yaml`
   - Builder + Executor（训练/报告）：在 YAML 里设置 `run_executor: true` 后同样运行
4) 验收（最小）
   - `state_save_path` 生成 `*.pkl`
   - 若启用 executor：生成 `report_path`（Markdown）与 `save/<case>/<timestamp>/` 目录产物

#### 0.2 2.5 目标链路（你要的：DAG 初始化 → 自动生成 TSPN 结构 → 微调/闭环优化）

目的：把“DAG 结构”变成 `model_config.yaml` 的来源之一，让外环更少依赖人工先验结构设计。

最小闭环（每轮）：
1) **Init DAG（数据驱动、确定性）**
   - 从 data_factory 读取数据后，创建 `ch1..chC` roots
   - 推断：`L/in_dim`、`C/in_channels`、`num_classes`
2) **BootstrapConfig（DAG→TSPN）**
   - 从 DAG 拓扑/算子语义生成初始 `model_config.yaml`（或 `ConfigPatch`）
3) **Smoke Run**
   - `epochs=1, batch=2, subset=10`（失败直接回退，返回 `INVALID_*`）
4) **Train / Fine-tune（内环）**
   - 对小 patch 优先 warm-start；否则从头训练
5) **Explain → Reflect**
   - 输出 `TrainReport`（metrics + explain_summary + reason_codes）
6) **ConfigPatch（外环）**
   - 生成下一轮 patch（增/删/改），进入下一轮

---

### Phase 0（P0）：对齐 Unified 结构的“静态可复现基线”

目标：在 PHMGA 中形成与 Unified 工程结构一一对应的模块，并用 `model_config.yaml` 控制结构，**人工给定配置即可跑通**（暂不引入外环搜索）。

1) 模块结构对齐（代码组织）
   - Unified：
     - `model/Signal_processing.py`
     - `model/Feature_extract.py`
     - `model/TSPN.py`
   - PHMGA（建议）：
     - `src/model/explainable/ops.py`（Signal_processing 对应）
     - `src/model/explainable/feature_ops.py`（Feature_extract 对应）
     - `src/model/explainable/tspn.py`（TSPN 对应）
     - `src/model/explainable/builder.py`（从 YAML build + op_uid manifest）
   - 验收：不安装 torch 时，import 这些模块不崩；只有真正 build/train 才 import torch（懒加载）。

2) 补齐 Unified feature tokens
   - 在 `feature_ops` 中实现 `config_basic.yaml` 中出现的全部 tokens（至少 13 个）
   - 验收：`make_feature(token)` 对每个 token 不抛异常；数值稳定（无 NaN）。

3) 补齐 Unified signal tokens（分 P0/P1 两步）
   - P0 必须：`I/WF/HT/FFT`（并满足 real + length-preserve contract）
   - P1 扩展：`Morlet/Laplace/*MA/*DF/Log/Squ/sin/LNO/add/mul/div`
   - 验收：每个 token 有单测覆盖其 I/O contract（shape、dtype、无 complex）。

4) 配置协议对齐（legacy adapter + 新 schema）
   - 支持直接读取 `Unified_X_fault_diagnosis/configs/config_basic.yaml`：
     - `signal_processing_configs` -> `layers[].ops[].token`
     - `feature_extractor_configs` -> `features[]`
     - `args.{in_dim,in_channels,num_classes,...}` -> `model/train` 字段
   - 同时提供“新口径 model_config.yaml”（只含 model/train/explain；不含 data_dir）
   - 验收：同一份 legacy 配置能被转换并落盘为规范 `model_config.yaml`（可复现）。

---

### Phase 1（P1）：内环训练/解释性证据产出（对齐 Unified 的可解释性）

目标：训练结束输出结构化证据（operator importance + WF 参数 + feature stats），为外环决策提供输入。

1) 训练器（内环）打通
   - 输入：`case.yaml + model_config.yaml`
   - 输出：`TrainReport` + artifacts（见 SPEC）
   - 验收：在装有 torch 的训练环境可跑通 1 个数据集（debug 模式先过）。

2) 解释性证据标准化
   - `operator_importance.json`：按 `op_uid` 输出 top-k + 全量 gate 值
   - `wavefilters_params.json`：导出 `fc/fb`（norm + 可选 Hz）
   - `feature_stats.json`：每个 feature token 的统计（mean/var/范围等）
   - 验收：Reporter 能引用这些证据生成可读报告段落。

---

### Phase 2（P2）：外环智能体自动生成/修改配置（减少先验）

目标：外环通过 `ConfigPatch` 自动“增删改查”网络结构，但实现方式必须是 **Immutable Config**（不热更新 nn.Module）。

0) DAG → TSPN 结构映射（必须先定死，避免“想当然”）
   - 输入：Builder 产物（DAG 拓扑 + 每个 node 的 `method/op_name` + 父子关系）
   - 输出：`model.layers[]`（每层并行 ops 列表）
   - 推荐的确定性 mapping（MVP）：
     - 定义 `dag_op_name -> tspn_token` 映射表：
       - `identity -> I`
       - `hilbert* -> HT`
       - `fft* -> FFT`
       - `wavefilter* -> WF`
       - `add/mul/div -> add/mul/div`（二元算子：要求输入通道数为 2C）
     - 以 DAG “深度”作为 layer index：
       - 深度 d（从 root=0 开始）出现的所有 token，组成 TSPN 的第 d+1 层的并行 ops
       - 对重复 token 通过 occurrence_idx 编号，生成稳定 `op_uid`
     - 对无法映射的 DAG op：
       - 作为“非可微预处理”保留在 DAG（不进入 TSPN），或在 mapping 时丢弃但记录到 `meta.unmapped_ops`
   - 验收：同一份 DAG 导出配置必须 determinisitc（同输入 → 同 YAML）。

0.1) Clarify `execute_agent` 职责（避免把“训练”塞进 DAG builder）
   - Builder graph 中的 `execute_agent`：只负责执行 `src/tools/*_schemas.py` 的“函数式 DAG 扩展”（旧模式）。
   - 训练/微调：放在 executor graph 的独立节点（例如 `deep_model_train_agent`），并输出结构化 `TrainReport`。
   - 若未来需要“训练型任务”在 builder 阶段介入，新增专用节点/agent（例如 `tspn_bootstrap_agent`），不要在 `execute_agent` 内引入大量分支与训练状态。

1) `model_config_agent`（生成 patch）
   - 输入：上一次 `model_config.yaml` + `TrainReport`（含 explain_summary）+ fail-fast 报错
   - 输出：`ConfigPatch`（严格白名单）
   - 关键策略：
     - “删”= `disabled_ops[op_uid]=1e-6` 或把某 op 从 layer list 移除（两者选其一，优先软删除）
     - “增/改”= 调整 layer token 列表、gate_temperature、WF init、feature token 列表、训练超参
   - 验收：输出能被 schema 校验并应用；任何非法 patch 都被拒绝并返回 `INVALID_*`。

2) `config_reflect_agent`（基于语义证据而非仅 metric）
   - 输入：confusion + WF 频带 + operator importance
   - 输出：`reason_codes`（例如高频混淆->建议增加高频滤波 token / 调整 WF init）
   - 验收：同一类错误模式能稳定给出相同 reason_codes（减少随机搜索感）。

3) 外层编排（Outer Graph）
   - `Plan -> Execute -> Reflect -> (Plan/Report)` 保持不变
   - 在每轮训练前强制 `Smoke Run`（AGENT_IO 规定）
   - 验收：闭环最多 N 轮（默认 5），每轮保存 `crud_history.json`，可追溯每次改了什么、为什么改。

---

### Phase 3（P3）：多数据集 + Few-shot DG（可选扩展）

目标：通过 case.yaml 任意切换数据集，并支持 Task C（少样本 DG）实验与消融。

1) data_factory 统一入口
   - 仅修改 `case.yaml` 的数据集名称即可切换（不影响 model_config）
2) episodic few-shot
   - `n_way/k_shot/q_query/n_episodes_*` 全流程可控
3) 消融实验模板
   - 结构消融：禁用某类 token（WF/HT/FFT）
   - 特征消融：feature token 子集
   - 搜索消融：无外环（固定 config） vs 外环闭环

验收：自动生成“主结果 + 消融 + 置信区间/重复次数”表格与结论段落。

---

### 交付清单（最小）

- 文档：`SPEC.md / AGENT_IO.md / PLAN.md`
- 代码（P0/P1）：算子/特征迁移 + 配置 schema + build/train/explain 产物
- 代码（P2）：外环 patch agent + reflect agent + smoke-run gate
- 测试：token I/O 合同单测 + config schema 校验单测 + import-time 无 torch/langgraph 崩溃
