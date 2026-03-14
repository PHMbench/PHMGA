# Execute Agent 深入分析

## 目录

1. [架构概览](#架构概览)
2. [输入输出契约](#输入输出契约)
3. [核心执行逻辑](#核心执行逻辑)
4. [Multi 算子处理](#multi-算子处理)
5. [Decision 算子处理](#decision-算子处理)
6. [错误处理与 Gap 记录](#错误处理与-gap-记录)
7. [测试覆盖分析](#测试覆盖分析)

---

## 架构概览

### 调用链路

```
WorkflowState (with step_plan)
    ↓
execute_agent() [src/agents/execute_agent.py]
    ├─→ render_execute_prompt() [渲染契约]
    ├─→ DAGTracker() [DAG 构建]
    ├─→ _existing_nodes() [恢复已有节点]
    ├─→ _seed_input_roots() [创建输入根节点]
    ├─→ 对每个 step:
    │   ├─→ catalog.get_by_plan_name() [查找算子]
    │   ├─→ llm.resolve_missing_params() [解析参数]
    │   ├─→ _execute_single() 或 _execute_multi() [执行]
    │   ├─→ tracker.add_node() [添加到 DAG]
    │   └─→ state.execution_results[node_id] = result [保存结果]
    └─→ tracker.export() [导出 DAG]
    ↓
WorkflowState (updated)
    ├─ dag: DagJson
    ├─ execution_results: Dict[str, np.ndarray]
    ├─ execution_gaps: List[ExecutionGap]
    └─ status: "executed"
```

### 关键文件

| 文件 | 职责 |
|------|------|
| [src/agents/execute_agent.py](../../src/agents/execute_agent.py) | Agent 入口，执行逻辑 |
| [src/prompts/execute_prompt.py](../../src/prompts/execute_prompt.py) | Prompt 模板和契约 |
| [src/operators/catalog.py](../../src/operators/catalog.py) | 算子目录实现 |
| [src/dag/ir.py](../../src/dag/ir.py) | DAG IR 定义 |

---

## 输入输出契约

### 输入 (WorkflowState)

```python
WorkflowState(
    user_instruction: str,
    dataset_name: str,
    graph_path: GraphPath,              # "dag_only" | "ml" | "torch"
    data_context: dict,
    signal_context: SignalContext,      # ✅ 必须已初始化
    step_plan: StepPlan,                # ✅ 必须已生成
    dag: Optional[DagJson] = None,      # 可能为空
)
```

**前置条件**:
- `signal_context` 不为 None
- `step_plan` 不为 None

### 输出 (更新的 WorkflowState)

```python
WorkflowState(
    ...
    dag: DagJson,                       # ✅ 新增/更新
    execution_results: Dict[str, np.ndarray],  # ✅ 新增/更新
    execution_gaps: List[ExecutionGap], # ✅ 重置
    status: "executed",                 # ✅ 更新
)
```

### ExecutionGap 结构

```python
ExecutionGap(
    step_index: int,                    # 失败的步骤索引
    parent: str,                        # 父节点
    op_name: str,                       # 算子名称
    message: str,                       # 错误信息
    recoverable: bool,                  # 是否可恢复
)
```

---

## 核心执行逻辑

### 1. 节点 ID 生成

```python
def _node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{parent.replace(',', '__')}"
```

**示例**:
- `normalize_01_ch1`
- `fft_03_normalize_01_ch1`
- `concatenate_11_rms_07_fft_03_normalize_01_ch1__rms_10_fft_04_normalize_02_ch2`

### 2. 节点类型推断

```python
def _node_kind(op_uid: str) -> str:
    if op_uid.startswith("feature."):
        return "feature"
    if op_uid.startswith("multi."):
        return "multi"
    if op_uid.startswith("decision."):
        return "decision"
    if op_uid.startswith("input."):
        return "input"
    return "transform"
```

### 3. 合法路径推断

```python
def _legal_paths(op_uid: str) -> List[str]:
    if op_uid in {"feature.mean", "feature.std"}:
        return ["dag_only", "ml"]
    if op_uid.startswith("decision."):
        return ["dag_only"]
    return ["dag_only", "ml", "torch"]
```

### 4. 输入根节点种子

```python
def _seed_input_roots(state: WorkflowState, protocol: DatasetProtocol, tracker: DAGTracker) -> None:
    sample_id, preview_window = materialize_preview_signal(protocol)
    state.signal_context.representative_sample_id = sample_id
    
    for channel_index, root_id in enumerate(state.signal_context.root_node_ids):
        if root_id in state.execution_results:
            continue  # 已存在，跳过
        
        channel_signal = np.asarray(preview_window[channel_index : channel_index + 1], dtype=float)
        tracker.add_node(
            DagNode(
                node_id=root_id,
                op_uid="input.signal",
                name=f"Input Channel {channel_index + 1}",
                kind="input",
                operator_category="input",
                params={"channel_index": channel_index},
                parents=[],
                in_shape=list(channel_signal.shape),
                out_shape=list(channel_signal.shape),
                backend_availability=["np", "pt", "sym"],
                execution_role="fixed",
                legal_paths=["dag_only", "ml", "torch"],
                plan_step_ref="root_input",
                rationale="Implicit multi-channel raw signal root.",
            )
        )
        state.execution_results[root_id] = channel_signal
```

**关键行为**:
- 为每个通道创建 input.signal 节点
- 将原始信号保存到 execution_results
- 如果已存在则跳过（支持增量执行）

---

## Multi 算子处理

### Multi 算子识别

```python
if operator.spec.op_uid.startswith("multi."):
    # 多输入算子
    result = _execute_multi(operator, parent_results, params)
    input_bindings = {f"arg{index}": parent_id for index, parent_id in enumerate(parent_ids)}
    in_shape = [sum(int(np.asarray(result_item).size) for result_item in parent_results)]
```

### Multi 执行逻辑

```python
def _execute_multi(op, parent_results: List[np.ndarray], params: Dict[str, float]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_results, **params), dtype=float)
```

### Concatenate 实现

```python
class ConcatenateOperator(BaseIsomorphicOperator):
    spec = OperatorSpec(
        op_uid="multi.concatenate",
        name="Concatenate",
        param_schema={"axis": "int"},
        input_shape_rule="1 + 1 + ...",
        output_shape_rule="K",
        backend_availability=["np", "sym"],
        execution_role="fixed",
    )

    def forward_np(self, x: List[np.ndarray], **kwargs: int) -> np.ndarray:
        axis = int(kwargs.get("axis", 0))
        return np.concatenate(x, axis=axis)
```

### Multi 节点示例

```python
DagNode(
    node_id="concatenate_11_rms_07_fft_03_normalize_01_ch1__rms_10_fft_04_normalize_02_ch2",
    op_uid="multi.concatenate",
    name="Concatenate",
    kind="multi",
    operator_category="multi",
    params={"axis": 0},
    parents=["rms_07_fft_03_normalize_01_ch1", "rms_10_fft_04_normalize_02_ch2"],
    in_shape=[2],                      # 父节点大小之和
    out_shape=[2],                     # 连接后的大小
    backend_availability=["np", "sym"],
    execution_role="fixed",
    legal_paths=["dag_only", "ml", "torch"],
    input_bindings={                   # ✅ 关键：参数绑定
        "arg0": "rms_07_fft_03_normalize_01_ch1",
        "arg1": "rms_10_fft_04_normalize_02_ch2"
    },
    plan_step_ref="step_11",
    rationale="Planner step 11: apply concatenate to rms_07...,rms_10... .",
)
```

---

## Decision 算子处理

### Decision 算子识别

```python
if operator.spec.op_uid.startswith("decision."):
    state.execution_gaps.append(
        ExecutionGap(
            step_index=step_index,
            parent=step.parent,
            op_name=step.op_name,
            message="Decision operators are auxiliary terminals and are not executed in the current runtime.",
            recoverable=True,
        )
    )
    continue  # 跳过执行
```

### 当前行为

| 状态 | 行为 |
|------|------|
| 识别到 decision.* | 记录为 recoverable gap |
| 不实际执行 | 跳过该步骤 |
| 继续执行 | 处理下一步 |

### 未来设计

**当前**（auxiliary terminal）:
```
decision 算子 → 记录 gap → 跳过
```

**目标**（runnable inner-loop op）:
```
decision 算子 → 执行分支逻辑 → 返回结果
```

---

## 错误处理与 Gap 记录

### Gap 类型

| 错误类型 | message | recoverable | 场景 |
|----------|---------|-------------|------|
| Missing parent | "Missing parent nodes: [...]" | False | 父节点不存在 |
| Unknown operator | "Unknown or unsupported operator: ..." | False | 算子不存在 |
| Param resolution | "Missing required parameter '...'" | False | 参数无法解析 |
| Multi-input error | "Single-input operator received multiple parents" | False | 单输入算子收到多个父节点 |
| Decision operator | "Decision operators are auxiliary terminals..." | True | Decision 算子（设计如此） |

### Gap 记录逻辑

```python
# 1. 检查父节点是否存在
missing_parents = [parent_id for parent_id in parent_ids if parent_id not in state.execution_results]
if missing_parents:
    state.execution_gaps.append(
        ExecutionGap(
            step_index=step_index,
            parent=step.parent,
            op_name=step.op_name,
            message=f"Missing parent nodes: {missing_parents}",
            recoverable=False,
        )
    )
    continue  # 跳过此步骤

# 2. 查找算子
try:
    operator = catalog.get_by_plan_name(step.op_name)
except KeyError as exc:
    state.execution_gaps.append(
        ExecutionGap(
            step_index=step_index,
            parent=step.parent,
            op_name=step.op_name,
            message=f"Unknown or unsupported operator: {exc}",
            recoverable=False,
        )
    )
    continue

# 3. 解析参数
try:
    params = llm.resolve_missing_params(...)
except ValueError as exc:
    state.execution_gaps.append(
        ExecutionGap(
            step_index=step_index,
            parent=step.parent,
            op_name=step.op_name,
            message=str(exc),
            recoverable=False,
        )
    )
    continue
```

---

## 测试覆盖分析

### 现有测试

#### test_execute_agent_materializes_results_and_multi_node

**覆盖**:
- ✅ execution_results 正确填充
- ✅ multi 节点正确生成
- ✅ 所有节点有 plan_step_ref 和 rationale

**不覆盖**:
- ❌ 错误处理逻辑
- ❌ decision 算子处理
- ❌ 增量执行（已有 DAG）

#### test_execute_agent_records_unknown_operator_gap_without_silent_fallback

**覆盖**:
- ✅ 未知算子记录为 gap
- ✅ 不静默失败（没有创建无效节点）

**不覆盖**:
- ❌ 其他 gap 类型
- ❌ recoverable vs unrecoverable
- ❌ gap 后的执行恢复

### 测试覆盖矩阵

| 场景 | 覆盖状态 |
|------|----------|
| 正常执行 | ✅ |
| Multi 节点 | ✅ |
| 未知算子 | ✅ |
| 缺失父节点 | ❌ |
| 参数解析失败 | ❌ |
| Decision 算子 | ❌ |
| 增量执行 | ❌ |
| 单算子多父节点 | ⚠️ (间接) |
| 边界情况 | ❌ |

---

## 总结

### 优点

- ✅ 严格的 step plan 消费，不发明新步骤
- ✅ 完整的错误记录（ExecutionGap）
- ✅ Multi 算子支持（concatenate）
- ✅ 节点元数据丰富（plan_step_ref, rationale）
- ✅ 支持增量执行（已有节点保留）

### 需要改进

- ⚠️ Decision 算子未真正执行（auxiliary terminal）
- ⚠️ 测试覆盖不完整（gap 类型、边界情况）
- ⚠️ 增量执行未测试
- ⚠️ 错误恢复策略未定义（gap 后是否继续）

### 优先级

| 优先级 | 改进项 | 复杂度 |
|--------|--------|--------|
| 高 | Decision 合同落地 | 高 |
| 高 | 补充 gap 类型测试 | 低 |
| 中 | 定义错误恢复策略 | 中 |
| 低 | 增量执行测试 | 中 |

---

## 相关文件

- [src/agents/execute_agent.py](../../src/agents/execute_agent.py)
- [src/prompts/execute_prompt.py](../../src/prompts/execute_prompt.py)
- [src/operators/catalog.py](../../src/operators/catalog.py)
- [src/dag/ir.py](../../src/dag/ir.py)
- [tests/unit/test_execute_agent.py](../../tests/unit/test_execute_agent.py)
