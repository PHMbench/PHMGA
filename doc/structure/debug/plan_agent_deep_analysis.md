# Plan Agent 深入分析

## 目录

1. [架构概览](#架构概览)
2. [输入输出契约](#输入输出契约)
3. [核心实现逻辑](#核心实现逻辑)
4. [Prompt 模板设计](#prompt-模板设计)
5. [测试覆盖分析](#测试覆盖分析)
6. [边界情况处理](#边界情况处理)
7. [潜在改进点](#潜在改进点)

---

## 架构概览

### 调用链路

```
WorkflowState
    ↓
plan_agent() [src/agents/plan_agent.py]
    ├─→ _build_signal_context() [首次]
    └─→ llm.generate_step_plan()
           └─→ OfflineLLM.generate_step_plan() [src/llm/client.py]
                  └─→ StepPlan (Pydantic)
    ↓
WorkflowState (updated)
    ├─ signal_context: SignalContext
    ├─ step_plan: StepPlan
    └─ status: "planned"
```

### 关键文件

| 文件 | 职责 |
|------|------|
| [src/agents/plan_agent.py](../../src/agents/plan_agent.py) | Agent 入口，状态管理 |
| [src/prompts/plan_prompt.py](../../src/prompts/plan_prompt.py) | Prompt 模板和契约 |
| [src/llm/client.py](../../src/llm/client.py) | LLM 接口和 offline stub |
| [src/states/workflow.py](../../src/states/workflow.py) | 状态定义 (SignalContext, StepPlan) |

---

## 输入输出契约

### 输入 (WorkflowState)

```python
WorkflowState(
    user_instruction: str,              # 用户指令
    dataset_name: str,                  # 数据集名称
    graph_path: GraphPath,              # "dag_only" | "ml" | "torch"
    data_context: dict,                 # {"min_depth", "min_width", "max_depth"}
    signal_context: Optional[SignalContext] = None,  # 首次时为 None
    step_plan: Optional[StepPlan] = None,           # 首次时为 None
)
```

### 输出 (更新的 WorkflowState)

```python
WorkflowState(
    ...
    signal_context: SignalContext,      # ✅ 新增
    step_plan: StepPlan,                # ✅ 新增
    status: "planned",                  # ✅ 更新
)
```

### SignalContext 结构

```python
SignalContext(
    dataset_name: str,                  # "RM_101_THU_GEARBOX"
    channel_count: int,                 # 8
    window_shape: List[int],            # [8, 4096]
    sampling_rate: int,                 # 10240
    source_mode: str,                   # "real" | "synthetic"
    root_node_ids: List[str],           # ["ch1", "ch2", ..., "ch8"]
    representative_sample_id: str,      # "101001"
    available_splits: List[str],        # ["train", "val", "test"]
)
```

### StepPlan 结构 (NVTA 风格)

```python
StepPlan(
    plan: [
        PlanStep(
            parent: str,                # "ch1" 或 "ch1,ch2" (多父)
            op_name: str,               # "normalize", "fft", "feature.mean"
            params: dict                # {"eps": 1e-6}
        ),
        ...
    ]
)
```

---

## 核心实现逻辑

### 1. SignalContext 构建

```python
def _build_signal_context(state: WorkflowState, protocol: DatasetProtocol) -> SignalContext:
    sample_id, preview_window = materialize_preview_signal(protocol)
    channel_count = int(preview_window.shape[0])
    return SignalContext(
        dataset_name=protocol.dataset_name,
        channel_count=channel_count,
        window_shape=list(preview_window.shape),
        sampling_rate=int(protocol.samples[0].sampling_rate),
        source_mode=protocol.source_mode,
        root_node_ids=[f"ch{index + 1}" for index in range(channel_count)],
        representative_sample_id=sample_id,
    )
```

**关键行为**:
- 使用 `materialize_preview_signal()` 获取预览窗口
- 生成标准化的根节点 ID: `ch1`, `ch2`, ...
- 提取采样率和窗口形状

### 2. DAG 深度计算

```python
def _dag_depth(state: WorkflowState) -> int:
    if not state.dag or not state.dag.nodes:
        return 0
    depth_by_node: dict[str, int] = {}
    for node in state.dag.nodes:
        if not node.parents:
            depth_by_node[node.node_id] = 1
        else:
            depth_by_node[node.node_id] = 1 + max(depth_by_node[parent] for parent in node.parents)
    return max(depth_by_node.values(), default=0)
```

**关键行为**:
- 空 DAG 返回 0
- 根节点深度为 1
- 每个子节点 = 1 + max(父节点深度)

### 3. OfflineLLM.generate_step_plan()

#### 初始规划逻辑 (DAG 为空)

```python
# 1. 每个通道添加 normalize
for root in roots:
    steps.append({"parent": root, "op_name": "normalize", "params": {"eps": 1e-6}})

# 2. 每个 normalize 后添加 fft
for index, root in enumerate(roots, start=1):
    steps.append({"parent": f"normalize_{index:02d}_{root}", "op_name": "fft", "params": {}})

# 3. 每个 fft 后添加特征 (mean, std, rms)
for index, root in enumerate(roots, start=1):
    fft_parent = f"fft_{len(roots) + index:02d}_normalize_{index:02d}_{root}"
    for feature_name in ("mean", "std", "rms"):
        steps.append({"parent": fft_parent, "op_name": feature_name, "params": {}})

# 4. ML/Torch 路径：添加 concatenate
if graph_path in {"ml", "torch"} and len(roots) > 1:
    rms_parents = [...]
    steps.append({"parent": ",".join(rms_parents), "op_name": "concatenate", "params": {"axis": 0}})
```

**生成的步骤序列 (2 通道)**:
```
1. ch1 → normalize
2. ch2 → normalize
3. normalize_01_ch1 → fft
4. normalize_02_ch2 → fft
5. fft_03_normalize_01_ch1 → mean
6. fft_03_normalize_01_ch1 → std
7. fft_03_normalize_01_ch1 → rms
8. fft_04_normalize_02_ch2 → mean
9. fft_04_normalize_02_ch2 → std
10. fft_04_normalize_02_ch2 → rms
11. rms_07...,rms_10... → concatenate (仅 ml/torch)
```

#### 增量规划逻辑 (DAG 非空)

```python
leaves = _leaf_node_ids(dag_json) or signal_context.root_node_ids
for leaf in leaves:
    if leaf.startswith("normalize_"):
        steps.append({"parent": leaf, "op_name": "fft", "params": {}})
    elif leaf.startswith("fft_"):
        steps.append({"parent": leaf, "op_name": "rms", "params": {}})
    else:
        steps.append({"parent": leaf, "op_name": "normalize", "params": {"eps": 1e-6}})
```

**关键行为**:
- 基于 DAG 叶子节点扩展
- 简单的前缀匹配策略
- 默认添加 normalize

---

## Prompt 模板设计

### 输入字段契约

```python
PLAN_PROMPT_INPUT_FIELDS = (
    "instruction",          # 用户指令
    "signal_context",       # 信号上下文
    "dag_json",            # 当前 DAG
    "tools",               # 算子目录
    "reflection",          # 反思历史
    "current_depth",       # 当前深度
    "min_depth",           # 最小深度目标
    "min_width",           # 最小宽度目标
)
```

### 输出字段契约

```python
PLAN_PROMPT_OUTPUT_FIELDS = ('{"plan": [{"parent": "...", "op_name": "...", "params": {...}}]}',)
```

### 禁止行为

```python
PLAN_PROMPT_PROHIBITIONS = (
    "invent operators that are not present in tools",
    "modify model or training parameters",
    "emit free-form prose instead of JSON",
)
```

### Prompt 模板结构

```
You are a world-class AI strategist specializing in PHM.

{contract}  # IO 契约 + 禁止行为

Strategic guidance:
1. Analyze the existing DAG before proposing the next steps.
2. If `dag_json` is empty, treat `signal_context.root_node_ids` as roots.
3. Prefer PHM workflows: time-domain → frequency-domain → features.
4. Use any existing node as a parent when logical.
5. Respect signal shapes. Don't apply aggregates to reduced features.

Rules:
- Return valid JSON only.
- Each plan item must contain `parent`, `op_name`, and `params`.
- The plan must remain executable by the operator catalog.

Instruction: {instruction}
Signal context: {signal_context}
Current DAG: {dag_json}
Available tools: {tools}
Reflection: {reflection}
Current depth: {current_depth}
Minimum depth: {min_depth}
Minimum width: {min_width}
```

---

## 测试覆盖分析

### 现有测试

#### test_plan_agent_outputs_nvta_style_step_plan

**覆盖**:
- ✅ signal_context 正确生成
- ✅ step_plan 符合 NVTA 格式
- ✅ root_node_ids 正确生成 (ch1, ch2, ...)
- ✅ 第一个步骤的 parent 以 "ch" 开头

**不覆盖**:
- ❌ 增量规划逻辑 (DAG 非空)
- ❌ 多通道 concatenate
- ❌ 边界情况 (空通道、单通道)

#### test_plan_prompt_contract_matches_agent_inputs_and_outputs

**覆盖**:
- ✅ Prompt 输入字段与 agent 一致
- ✅ Prompt 输出字段包含 "plan"
- ✅ Prompt 包含关键信息 ("Signal context", "Current DAG")

**不覆盖**:
- ❌ 禁止行为的验证
- ❌ 实际渲染的 prompt 内容验证

### 测试覆盖矩阵

| 场景 | 覆盖状态 |
|------|----------|
| 初始规划 (空 DAG) | ✅ |
| 增量规划 (非空 DAG) | ❌ |
| 单通道 | ✅ |
| 多通道 (>2) | ⚠️ (间接) |
| dag_only 路径 | ✅ |
| ml 路径 | ⚠️ (间接) |
| torch 路径 | ⚠️ (间接) |
| concatenate 生成 | ⚠️ (间接) |
| 边界情况 (空通道) | ❌ |
| 边界情况 (超多通道) | ❌ |

---

## 边界情况处理

### 当前实现

| 边界情况 | 当前行为 | 是否合理 |
|----------|----------|----------|
| 空 DAG | 生成初始规划 | ✅ |
| 单通道 | 不生成 concatenate | ✅ |
| 零通道 (channel_count=0) | root_node_ids=[] | ⚠️ 静默失败 |
| 超多通道 (>100) | 生成大量步骤 | ⚠️ 无限制 |
| 无效 graph_path | 忽略，使用默认逻辑 | ⚠️ 无验证 |
| 缺少 data_context | 使用默认值 (min_depth=2, min_width=1) | ✅ |

### 潜在问题

1. **零通道情况**
   ```python
   # 如果 channel_count = 0
   root_node_ids = []  # 空列表
   # 后续步骤生成会跳过，但不会报错
   ```

2. **超多通道情况**
   ```python
   # 如果 channel_count = 100
   # 会生成 100 + 100 + 300 + 1 = 501 步 (ml 路径)
   # 可能导致性能问题
   ```

3. **graph_path 验证**
   ```python
   # 当前只检查 {"ml", "torch"}
   # 无效路径会被静默忽略
   ```

---

## 潜在改进点

### 1. 增加边界情况测试

```python
def test_plan_agent_handles_single_channel():
    """单通道不应生成 concatenate"""

def test_plan_agent_validates_graph_path():
    """无效 graph_path 应报错"""

def test_plan_agent_handles_empty_channels():
    """零通道应返回空 plan 或报错"""
```

### 2. 改进增量规划策略

当前简单的前缀匹配不够健壮：
```python
# 当前
if leaf.startswith("normalize_"):
    ...

# 建议：基于算子类别
if any(n.op_uid.startswith("normalize.") for n in dag.nodes if n.node_id == leaf):
    ...
```

### 3. 添加规划深度限制

```python
MAX_PLAN_STEPS = 100
if len(steps) > MAX_PLAN_STEPS:
    # 警告或限制
```

### 4. 改进 error messages

```python
# 当前
raise ValueError("signal_context must be initialized")

# 建议
raise ValueError(
    "signal_context must be initialized before execute_agent runs. "
    "Call plan_agent first."
)
```

---

## 总结

### 优点

- ✅ 清晰的 NVTA 风格契约
- ✅ Pydantic 模型确保类型安全
- ✅ Prompt 模板结构化良好
- ✅ Offline stub 可测试

### 需要改进

- ⚠️ 边界情况处理不足
- ⚠️ 测试覆盖不完整
- ⚠️ 增量规划策略简单
- ⚠️ 缺少输入验证

### 优先级

| 优先级 | 改进项 | 复杂度 |
|--------|--------|--------|
| 高 | 增加边界情况测试 | 低 |
| 高 | 添加 graph_path 验证 | 低 |
| 中 | 改进增量规划策略 | 中 |
| 低 | 添加规划深度限制 | 低 |

---

## 相关文件

- [src/agents/plan_agent.py](../../src/agents/plan_agent.py)
- [src/prompts/plan_prompt.py](../../src/prompts/plan_prompt.py)
- [src/llm/client.py](../../src/llm/client.py)
- [src/states/workflow.py](../../src/states/workflow.py)
- [tests/unit/test_plan_agent.py](../../tests/unit/test_plan_agent.py)
