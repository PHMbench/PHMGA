# Agents Structure

## 职责
- `agents` 层承载 outer workflow 中可独立调度的业务节点。
- 它负责把 `PHMState`、research state、tools、model、llm 连接成单步动作。
- 它不负责 Hydra compose、case 运行目录规划、算子注册定义，也不负责底层模型实现。

## 为什么需要这一层
- builder/executor/report/train/research 都是长流程；没有独立节点层，runner 和 graph 会被业务细节淹没。
- `agent` 在本仓库的定义是“outer workflow node”，不是“必须调用大模型的组件”。
- deterministic 节点继续保留 `agent` 命名，是为了保持 workflow 语义稳定；如果为了名字纯洁再拆出 `step/helper/worker`，只会增加偶然复杂性。

## 正式入口
- 当前正式实现入口仍是平铺文件：
  - `src/agents/plan_agent.py`
  - `src/agents/execute_agent.py`
  - `src/agents/reflect_agent.py`
  - `src/agents/report_agent.py`
  - `src/agents/dag_init_agent.py`
  - `src/agents/dataset_preparer_agent.py`
  - `src/agents/deep_model_train_agent.py`
  - `src/agents/inquirer_agent.py`
  - `src/agents/shallow_ml_agent.py`
  - `src/agents/tspn_bootstrap_agent.py`
  - `src/agents/deep_research_agents.py`
- `src/agents/builder`、`executor`、`report`、`train` 当前多数只是 re-export 兼容壳，不是正式逻辑归宿。
- `src/agents/shared/compat.py` 是过渡兼容层，不承载业务逻辑。

## 分类原则
### LLM-mediated agents
- 节点的核心决策依赖 LLM 输出，或通过 prompt/structured output 生成业务决策。
- 典型特征：调用 `get_llm()`、依赖 prompt、输出带有“规划/反思/报告/研究”语义。

### Deterministic agents
- 节点是 outer workflow 的正式步骤，但核心逻辑不依赖 LLM。
- 典型特征：读取 state/dataset/config，执行确定性数据处理、训练、bootstrap 或相似度计算。

## Agent I/O 合同表
### LLM-mediated agents
#### `plan_agent`
- 职责：根据用户指令、当前 DAG 拓扑和历史反思生成 `detailed_plan`。
- 是否依赖 LLM：是。
- 正式入口：`src/agents/plan_agent.py::plan_agent`
- 输入：
  - `PHMState.user_instruction`
  - `PHMState.reflection_history`
  - `PHMState.dag_state.nodes`
  - `PHMState.min_depth` / `min_width` / `max_depth`
  - `PHMState.data_cfg.operator_contract`
  - `PHMState.data_cfg.enforce_tspn_closed_world`
- 输出：
  - 返回 `{"detailed_plan": List[dict]}`
  - 每个 step 至少包含 `parent`、`op_name`、`params`
- 状态副作用：
  - 可能向 `state.error_logs` 追加 planner sanitize / parser error
- 失败方式：
  - LLM 响应无法解析时返回空 `detailed_plan`
  - 不允许的算子会在 sanitize 阶段被丢弃并记录 warning
- 当前地位：builder 主规划节点，稳定。
- 后续收口方向：保留单一正式入口；不再为 planner 增加新的 wrapper。

#### `dag_init_agent`
- 职责：用 LLM 为每个 channel 生成最小预处理链，建立 bootstrap 友好的初始 `ProcessedData` DAG。
- 是否依赖 LLM：是。
- 正式入口：`src/agents/dag_init_agent.py::dag_init_agent`
- 输入：
  - `PHMState.dag_state.channels`
  - channel root `InputData.results["train"]`
  - channel root `meta.fs`
  - `max_ops_per_channel`
  - `temperature`
- 输出：
  - 返回 `{"dag_state": DAGState}`
  - 新 DAG 的 leaves 指向 `init_*` 处理节点
- 状态副作用：
  - 无就地修改；返回新的 `dag_state`
- 失败方式：
  - LLM 失败时回退到每个 channel 至少一层 `fft`
  - root 不是 `InputData` 或 train split 缺失时推断 `L/fs` 会退化为空值
- 当前地位：executor 中 TSPN fast path 的 LLM bridge。
- 后续收口方向：保留最小初始化职责，不扩展为通用 planner。

#### `execute_agent`
- 职责：执行 `detailed_plan`，扩展 DAG；在 `neuro_symbolic_train` 模式下路由到 bootstrap + deep train。
- 是否依赖 LLM：部分依赖。
  - 正常 DAG 执行时只在缺少必填算子参数时调用 LLM 补参数。
  - `neuro_symbolic_train` 路径本身是 deterministic route。
- 正式入口：`src/agents/execute_agent.py::execute_agent`
- 输入：
  - `PHMState.detailed_plan`
  - `PHMState.dag_state`
  - `PHMState.task_type`
  - `PHMState.data_cfg.operator_contract`
  - `PHMState.data_cfg.enforce_tspn_closed_world`
- 输出：
  - DAG 模式：`{"dag_state": DAGState, "executed_steps": int}`
  - neuro-symbolic 模式：训练相关更新，至少包含 `ml_results` / `run_dir`，必要时带 `model_config_path`
- 状态副作用：
  - 可能向 `state.dag_state.error_log` 追加执行错误
  - 会写中间 `.npy/.npz` 节点产物和 DAG PNG/DOT
- 失败方式：
  - contract violation 立即停止当前执行轮
  - 参数生成失败、算子运行失败、shape 不匹配会写入 error log 并中断后续步骤
- 当前地位：builder 主执行节点，同时保留一种训练直达路由。
- 后续收口方向：后续可把 train route 从执行器中分离，但本轮不动。

#### `reflect_agent`
- 职责：审视当前 DAG 和错误上下文，决定 finish / patch / replan / halt。
- 是否依赖 LLM：是。
- 正式入口：
  - 业务 helper：`src/agents/reflect_agent.py::reflect_agent`
  - graph adapter：`src/agents/reflect_agent.py::reflect_agent_node`
- 输入：
  - `PHMState.user_instruction`
  - `PHMState.dag_state`
  - `PHMState.min_depth` / `min_width` / `max_depth`
  - `PHMState.dag_state.error_log`
  - adapter 额外接收 `stage`
- 输出：
  - helper 返回 `{"decision": str, "reason": str}`
  - node adapter 返回 `{"needs_revision": bool, "reflection_history": List[str], "iteration_count": int}`
- 状态副作用：
  - 无就地修改；通过返回增量驱动 builder loop
- 失败方式：
  - 输入缺失或 LLM 解析失败时返回 `halt`
- 当前地位：builder loop 的正式 gate。
- 后续收口方向：保留 helper + node adapter 双层，但文档只把 `reflect_agent_node` 视为 graph 正式入口。

#### `report_agent`
- 职责：基于 DAG、相似度统计和训练结果生成最终报告；LLM 不可用时退回模板报告。
- 是否依赖 LLM：是，但带 deterministic fallback。
- 正式入口：
  - 业务 helper：`src/agents/report_agent.py::report_agent`
  - graph adapter：`src/agents/report_agent.py::report_agent_node`
- 输入：
  - `PHMState.user_instruction`
  - `PHMState.dag_state`
  - `PHMState.ml_results`
  - leaf `node.sim`
  - `PHM_REPORT_MODE` / `FAKE_LLM`
- 输出：
  - helper 返回 `{"report_markdown": str}`
  - node adapter 返回 `{"final_report": str}`
- 状态副作用：
  - 会导出最终 DAG PNG/DOT
- 失败方式：
  - LLM 调用失败时回退到 `_template_report`
- 当前地位：executor/report 终点节点。
- 后续收口方向：继续保持 node adapter 为 graph 正式入口。

#### `deep_research_agents`
- 职责：独立 research 子图，用于 query generation、web research、reflection、finalize answer。
- 是否依赖 LLM：是。
- 正式入口：
  - module：`src/agents/deep_research_agents.py`
  - node functions：`generate_query`、`web_research`、`reflection`、`evaluate_research`、`finalize_answer`
- 输入：
  - `OverallState` / `QueryGenerationState` / `ReflectionState` / `WebSearchState`
  - `RunnableConfig`
  - `Configuration.from_runnable_config(config)`
- 输出：
  - research state 增量，例如 `search_query`、`web_research_result`、`is_sufficient`、`messages`
- 状态副作用：
  - 无 repo 内状态副作用；依赖外部搜索和 LLM provider
- 失败方式：
  - Google grounding 不可用时回退到普通 LLM web summary
- 当前地位：独立的 research agent 族，不属于 `PHMState` 主链。
- 后续收口方向：单独收口到 research graph，不和主 PHM workflow 混写。

### Deterministic agents
#### `dataset_preparer_agent`
- 职责：从 processed nodes 的保存产物构建 `datasets`，并为每个数据集创建 `DataSetNode`。
- 是否依赖 LLM：否。
- 正式入口：`src/agents/dataset_preparer_agent.py::dataset_preparer_agent`
- 输入：
  - `PHMState.dag_state.nodes`
  - processed node `meta.saved.train_path` / `val_path` / `test_path`
  - root `meta.labels_train` / `labels_val` / `labels_test`
  - `config.stage`
  - `config.flatten`
- 输出：
  - `{"datasets": Dict[str, Dict[str, Any]], "n_nodes": int}`
  - dataset entries 包含 `X_train/X_val/X_test/y_train/y_val/y_test`
- 状态副作用：
  - 通过 tracker 向 DAG 添加 `DataSetNode`
  - 可能向 `state.dag_state.error_log` 追加根标签追溯错误
- 失败方式：
  - 根节点缺标签、特征文件缺失或为空时跳过该节点
- 当前地位：executor 中 dataset 组装节点，稳定。
- 后续收口方向：保持 deterministic；不包装成 LLM agent。

#### `deep_model_train_agent`
- 职责：训练 torch-side TSPN，并写出 metrics、preflight、compile/compatibility 等工件。
- 是否依赖 LLM：否。
- 正式入口：`src/agents/deep_model_train_agent.py::deep_model_train_agent`
- 输入：
  - `PHMState.model_config_path`
  - `PHMState.model_cfg`
  - `PHMState.data_cfg`
  - `PHMState.labels_train` / `labels_val` / `labels_test`
  - channel root `InputData.results["train"|"val"|"test"]` 或 vibench data factory
  - 可选 `config` 覆盖训练参数
- 输出：
  - 至少返回：
    - `ml_results`
    - `run_dir`
    - `train_history`
    - `current_model_config`
    - `model_config_path`
- 状态副作用：
  - 写出 run directory、metrics、manifest、preflight、compatibility 报告等
- 失败方式：
  - 缺 `state.model_config_path`、模型配置文件不存在、root `results["train"]` 缺失时 fail fast
  - model/data dims 不匹配时显式报错
- 当前地位：executor/train 主训练节点，稳定。
- 后续收口方向：继续把模型真源固定在 `state.model_config_path` 和 `state.model_cfg`。

#### `tspn_bootstrap_agent`
- 职责：不依赖 LLM，依据已构建 DAG 生成最小可训练的 TSPN `model_config.yaml`。
- 是否依赖 LLM：否。
- 正式入口：`src/agents/tspn_bootstrap_agent.py::tspn_bootstrap_agent`
- 输入：
  - `PHMState.dag_state`
  - `PHMState.labels_train`
  - bootstrap 参数：`max_layers`、`parallel_ops_per_layer`、`out_channels`、`scale`、`features`
- 输出：
  - `{"model_config_path": str, "current_model_config": dict}`
- 状态副作用：
  - 写 bootstrap yaml 到 case save 目录
- 失败方式：
  - DAG 无 channel roots、有环、train split 缺失、shape 非 `(1, L, 1)` 时显式报错
- 当前地位：TSPN fast path 的 deterministic config bootstrap。
- 后续收口方向：保持 small deterministic bootstrap，不承载 bridge 策略。

#### `inquirer_agent`
- 职责：在 leaf nodes 上计算相似度，支持 canonical single-node split layout 和 paired leaves layout。
- 是否依赖 LLM：否。
- 正式入口：`src/agents/inquirer_agent.py::inquirer_agent`
- 输入：
  - `PHMState.dag_state.leaves`
  - leaf `results["train"]` / `results["test"]`
  - `metrics: List[str]`
- 输出：
  - `{"new_nodes": List[str]}`
  - paired layout 下会创建 similarity nodes；canonical layout 下主要通过 `node.sim` 暴露结果
- 状态副作用：
  - 会就地填充 leaf `node.sim`
  - paired layout 会向 DAG 添加 similarity nodes
  - shape mismatch 等问题写入 `state.dag_state.error_log`
- 失败方式：
  - 未知 metric 直接抛错
  - 数据缺失或 shape 不匹配时跳过对应 pair
- 当前地位：executor 中 similarity 节点，稳定。
- 后续收口方向：保留 deterministic agent 定义，不拆成 utility。

#### `shallow_ml_agent`
- 职责：对 prepared datasets 训练浅层模型并生成 ensemble 结果。
- 是否依赖 LLM：否。
- 正式入口：`src/agents/shallow_ml_agent.py::shallow_ml_agent`
- 输入：
  - `datasets: Dict[str, Dict[str, Any]]`
  - `algorithm`
  - `ensemble_method`
  - `cv_folds`
- 输出：
  - `{"models": ..., "ensemble_metrics": ..., "metrics_markdown": ...}`
- 状态副作用：
  - 无 `PHMState` 副作用；仅返回结果
- 失败方式：
  - `pandas` 不可用或 datasets 为空时返回空结果而非异常
- 当前地位：executor 中浅层基线训练节点。
- 后续收口方向：如需更严格合同，可后续在 caller 层显化空数据早停。

## 当前实现状态
- 平铺 `src/agents/*.py` 仍是实际实现和正式入口。
- `reflect_agent`、`report_agent` 已形成“业务 helper + graph adapter”双入口；graph 语义以 node adapter 为正式入口。
- `deep_research_agents.py` 不是单一 `PHMState` agent，而是独立 research 子图的 node 集合。
- `builder`、`executor`、`report`、`train` 子目录当前多数只承担兼容 re-export，不应继续扩展逻辑。

## 冗余与历史包袱
- 平铺入口和子目录 facade 并存，容易让新逻辑继续塞进兼容壳。
- `src/agents/shared/compat.py` 目前只是过渡透传层，没有长期抽象价值。
- `src/agents/__pycache__` 及各子目录 `__pycache__` 属于构建产物，不属于结构的一部分。
- `src/agents/readme.md` 和本结构文档存在职责重叠；前者应保留用法和架构概览，后者负责正式边界与合同。

## 按 v1.0 的下一步
- 在不改变 workflow 语义的前提下，先用测试锁定每个 agent 的 I/O 合同，再决定是否物理迁移到分目录。
- 子目录 facade 的删除条件固定为：
  - 旧导入路径不再被引用
  - 结构测试已锁定正式入口
  - `src/agents/readme.md` 和 `doc/structure/agents/structure.md` 已完成口径收口
- 不为 deterministic 节点再造 `step/helper/worker` 抽象层；真正需要拆分时，只拆局部 helper，不改 outer workflow node 语义。
