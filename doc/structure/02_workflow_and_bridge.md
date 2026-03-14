# 02 Workflow And Bridge

## 本文档解决什么问题

本文档冻结论文版前端与后端之间的正式边界：

1. 四个主路径 agent 的输入、输出和禁止事项
2. prompt 合同与 agent 输入输出的一致关系
3. `validated DAG JSON` 为什么仍然是唯一法定中间接口

当前参考资产来自两处：

- `feature-NSNet`
- `/home/user/LQ/C_Agent/PHMGA`

它们提供的是研究资产和 prompt 分工思想，不是整包复制对象。

## 当前实现态

前端运行时合同已经从旧的占位文本收敛到：

`signal_context + current_dag + reflection -> StepPlan -> execute_agent -> validated DAG JSON -> bridge`

这里有三个关键决定：

1. `graph_path` 由 config/runtime 决定，不作为 `plan_agent` 的显式 prompt 输入字段
2. 运行时主合同是 `StepPlan`，不是大而全的 `PlanSpec`
3. `execute_agent` 必须把代表性计算结果写回 state，而不是只生成一个空 DAG 壳

## 参考资产如何吸收

### 来自 `/home/user/LQ/C_Agent/PHMGA/src/prompts/plan_prompt.py`

保留：

- 单轮围绕现有 DAG 做结构扩展
- 允许从任意现有节点继续扩图
- 强调 PHM 域时域、频域、时频、跨通道思路
- 输出 `plan: [{parent, op_name, params}]`

重写：

- 不把 `graph_path` 放入 planner 的显式输入
- 不让 planner 直接越权影响 `model/`、`training/`
- 不把 prompt 写成旧平台式自治系统

### 来自 `/home/user/LQ/C_Agent/PHMGA/src/prompts/execute_prompt.py`

只保留：

- 读取 plan、DAG、tools 的输入组织方式

明确删除：

- executor 自己决定“下一步工具”的自治逻辑
- 从 leaves 自主规划新步骤的能力

### 来自 `/home/user/LQ/C_Agent/PHMGA/src/prompts/reflect_prompt.py`

保留：

- 结构合法性审查
- 深度/宽度/冗余/目标导向性审查
- `decision/reason` 这一对 NVTA 风格输出

补强：

- 额外输出 `missing_operators`
- 额外输出 `shape_risks`
- 额外输出 `structural_warnings`

### 来自 `/home/user/LQ/C_Agent/PHMGA/src/prompts/report_prompt.py`

保留：

- 报告要基于证据而不是自由发挥

重写：

- 正式输入改为 `graph-dependent artifacts + reflection summary`
- review context 作为辅输入保留
- 不再把旧平台的相似度/模型段落当作唯一模板

## 正式 prompt 文件树

```text
src/prompts/
├─ __init__.py
├─ shared.py
├─ plan_prompt.py
├─ execute_prompt.py
├─ reflect_prompt.py
├─ report_prompt.py
├─ inquirer_prompt.py      # optional
└─ reflector_prompt.py     # reference-only
```

`templates.py` 只保留兼容转发，不再是正式 prompt 入口。

## 运行时合同类型

### `SignalContext`

`plan_agent` 不直接接收完整窗口张量，而是接收 prompt-safe 的信号摘要：

```python
class SignalContext:
    dataset_name: str
    channel_count: int
    window_shape: list[int]
    sampling_rate: int
    source_mode: str
    root_node_ids: list[str]
    representative_sample_id: str | None
```

语义：

- 如果 `dag_json` 为空，则 `root_node_ids` 就是初始多通道信号根
- 这满足“没有初始节点就是多通道信号”的要求

### `StepPlan`

当前运行时的 planner 正式输出是：

```python
class PlanStep:
    parent: str
    op_name: str
    params: dict[str, Any]


class StepPlan:
    plan: list[PlanStep]
```

说明：

- `parent` 可以是单父节点 ID，也可以是逗号分隔的多父节点 ID
- `op_name` 使用 NVTA 风格紧凑名称，例如 `normalize`, `fft`, `mean`, `concatenate`
- `params` 只保存该 step 明确提供的参数，缺失参数由 execute 阶段补全

### `ExecutionGap`

```python
class ExecutionGap:
    step_index: int
    parent: str
    op_name: str
    message: str
    recoverable: bool
```

语义：

- `execute_agent` 不能静默跳过非法或无法执行的 step
- 无法满足的 step 必须显式写成 `ExecutionGap`

### `ReflectionResult`

```python
class ReflectionResult:
    decision: Literal["finish", "need_patch", "need_replan", "halt"]
    reason: str
    missing_operators: list[str]
    shape_risks: list[str]
    structural_warnings: list[str]
```

说明：

- `decision/reason` 与 NVTA 风格兼容
- 其余字段是论文版为结构审查补上的显式证据

## 四个 agent 的正式输入输出

### `plan_agent`

显式输入：

- `user_instruction`
- `signal_context`
- `dag_json`
- `reflection`

隐式上下文：

- `graph_path` from config/runtime
- `operator_catalog_summary`
- `protocol_summary`

输出：

- `StepPlan`

禁止事项：

- 不得输出 catalog 中不存在的 `op_name`
- 不得直接生成训练器参数
- 不得把自然语言说明冒充为正式 plan

### `execute_agent`

显式输入：

- `step_plan`
- `workflow_state`
- `operator_catalog`
- `signal_context`

隐式上下文：

- `graph_path` from config/runtime

输出：

- `validated DAG JSON`
- `execution_results` 写回 state
- `execution_gaps` 写回 state

必须遵守：

1. 只能消费 `StepPlan`
2. 先找 parent，再找 operator，再补参数，再执行
3. 参数补全顺序固定为：
   - state/protocol/context
   - operator schema 默认值
   - LLM 补全缺失必需参数
4. 无法执行时必须写 `ExecutionGap`
5. 不能直接写磁盘 artifact，文件落盘留给 `scripts/run_case.py`

### `reflect_agent`

显式输入：

- `instruction`
- `stage`
- `dag_blueprint`
- `issues_summary`
- `min_depth`
- `min_width`
- `max_depth`
- `current_depth`

输出：

- `ReflectionResult`

禁止事项：

- 不得直接 patch DAG
- 不得用训练指标替代结构合法性判断

### `report_agent`

核心输入：

- `instruction`
- `graph_path`
- `compiled_manifest`
- `path_artifacts`
- `reflection_summary`

兼容 review context：

- `stage`
- `dag_blueprint`
- `issues_summary`
- `min_depth`
- `min_width`
- `max_depth`
- `current_depth`

输出：

- `report_markdown`

说明：

- 这允许报告既遵循 graph-dependent artifacts，又兼容 NVTA 风格的 review context

## prompt 与 agent 的一一对应

### `plan_prompt.py`

输入字段：

- `instruction`
- `signal_context`
- `dag_json`
- `tools`
- `reflection`
- `current_depth`
- `min_depth`
- `min_width`

输出字段：

- `{"plan": [{"parent": "...", "op_name": "...", "params": {...}}]}`

对应 agent：

- `plan_agent`

### `execute_prompt.py`

输入字段：

- `step_plan`
- `dag_json`
- `operator_catalog`
- `signal_context`
- `graph_path`

输出字段：

- `{"node_updates": [...], "execution_gaps": [...]}`

对应 agent：

- `execute_agent`

### `reflect_prompt.py`

输入字段：

- `instruction`
- `stage`
- `dag_blueprint`
- `issues_summary`
- `min_depth`
- `min_width`
- `max_depth`
- `current_depth`

输出字段：

- `{"decision": "...", "reason": "...", "missing_operators": [], "shape_risks": [], "structural_warnings": []}`

对应 agent：

- `reflect_agent`

### `report_prompt.py`

输入字段：

- `instruction`
- `graph_path`
- `compiled_manifest`
- `path_artifacts`
- `reflection_summary`
- `review_context`

输出字段：

- markdown report

对应 agent：

- `report_agent`

## `graph_path` 的正式位置

`graph_path` 只应存在于：

- `config`
- `runtime_config`
- `WorkflowState.graph_path`
- bridge/path-specific backend

它不应重复作为 `plan_agent` 的显式 prompt 输入字段。

原因：

- 前端默认先生成 DAG
- 后端再根据 `graph_path` 决定是 `dag_only`、`ml` 还是 `torch`

## `execute_agent` 与 state 持久化

当前论文版明确要求：

- `execute_agent` 必须把代表性计算结果写回 state
- state 中至少保留：
  - 根输入结果
  - 新节点结果
  - `execution_gaps`

但它不负责：

- `dag.json` 落盘
- manifest 落盘
- final report 落盘

这些属于外层脚本与 evaluation/artifact 责任。

## bridge 的法定边界

以下边界必须保持不变：

- workflow/front-end 世界只输出 `validated DAG JSON`
- bridge 只消费 `validated DAG JSON`
- agents 不得直接把 state 塞给 `model/` 或 `training/`

正式链路：

`states / prompts / agents -> DAGTracker(NetworkX) -> DAG JSON -> validated DAG JSON -> bridge -> {dag_only | ml | torch}`

## 当前缺口与推荐默认

### 仍未完全实现的点

- provider-backed OpenRouter 真实生成仍未接入主测试
- `decision` 仍是 schema-first / auxiliary terminal
- richer multi-operator 仍未大规模迁移

### 推荐默认

1. 先稳住 `StepPlan -> execute -> validated DAG JSON`
2. 再扩更丰富的算子和 provider-backed prompting
3. 始终保持 prompt 字段、agent 输入输出、测试断言三者同名同义
