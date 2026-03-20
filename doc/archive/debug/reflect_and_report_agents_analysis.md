# Reflect & Report Agents 深入分析

> Historical note
>
> 本文写于前端回迁到 `PHMState + LangChain-style agents + LangGraph StateGraph` 之前。文中的 `WorkflowState` 和旧反射/报告调用链保留为迁移前分析，不代表当前实现。当前权威说明见 `README.md` 与 `doc/structure/02_workflow_and_bridge.md`。

## 目录

1. [Reflect Agent](#reflect-agent)
2. [Report Agent](#report-agent)

---

## Reflect Agent

### 架构概览

#### 调用链路

```
WorkflowState (after execute)
    ↓
reflect_agent() [src/agents/reflect_agent.py]
    ├─→ render_reflect_prompt() [渲染契约]
    ├─→ _dag_depth() [计算当前深度]
    ├─→ llm.reflect_workflow() [决策]
    └─→ state.reflection_results.append(result)
    ↓
WorkflowState (updated)
    ├─ reflection_results: List[ReflectionResult]
    ├─ reflection_history: List[str]
    └─ status: "reflected"
```

### 输入输出契约

#### 输入 (WorkflowState)

```python
WorkflowState(
    ...
    dag: DagJson,                       # ✅ 应该已执行
    execution_gaps: List[ExecutionGap],
    data_context: dict,                 # 包含 min_depth, min_width, max_depth
)
```

#### 输出 (更新的 WorkflowState)

```python
WorkflowState(
    ...
    reflection_results: List[ReflectionResult],  # ✅ 新增
    reflection_history: List[str],               # ✅ 新增
    status: "reflected",                         # ✅ 更新
)
```

### ReflectionResult 结构

```python
ReflectionResult(
    decision: ReflectionDecision,       # "finish" | "need_patch" | "need_replan" | "halt"
    reason: str,                        # 决策理由
    missing_operators: List[str],       # 缺失算子
    shape_risks: List[str],             # 形状风险
    structural_warnings: List[str],     # 结构警告
)
```

### 决策逻辑

#### OfflineLLM.reflect_workflow()

```python
def reflect_workflow(
    self,
    *,
    instruction: str,
    stage: str,
    dag_blueprint: Dict[str, Any],
    issues_summary: str,
    min_depth: int,
    min_width: int,
    max_depth: int,
    current_depth: int,
    execution_gaps: List[ExecutionGap],
) -> ReflectionResult:
    # 1. 检查 DAG 是否为空
    if not dag_blueprint.get("nodes"):
        return ReflectionResult(
            decision="halt",
            reason="DAG blueprint is empty.",
            structural_warnings=["No nodes were materialized."],
        )
    
    # 2. 检查执行间隙
    if execution_gaps:
        missing_operators = sorted({
            gap.op_name for gap in execution_gaps 
            if "Unknown" in gap.message or "unsupported" in gap.message.lower()
        })
        shape_risks = [gap.message for gap in execution_gaps if "shape" in gap.message.lower()]
        return ReflectionResult(
            decision="need_replan",
            reason=issues_summary or "Execution gaps prevent the current plan from completing cleanly.",
            missing_operators=missing_operators,
            shape_risks=shape_risks,
            structural_warnings=[gap.message for gap in execution_gaps],
        )
    
    # 3. 检查深度是否满足
    if current_depth < min_depth:
        return ReflectionResult(
            decision="need_patch",
            reason=f"The workflow is healthy but depth {current_depth} is below the minimum target {min_depth}.",
            structural_warnings=["Continue expanding the DAG."],
        )
    
    # 4. 一切正常
    return ReflectionResult(
        decision="finish",
        reason="The DAG is structurally valid and satisfies the current planning target.",
        missing_operators=[],
        shape_risks=[],
        structural_warnings=[],
    )
```

### 决策树

```
DAG 为空？
├─ Yes → "halt"
└─ No → 继续检查
        ↓
有 execution_gaps？
├─ Yes → "need_replan"
└─ No → 继续检查
        ↓
current_depth < min_depth？
├─ Yes → "need_patch"
└─ No → "finish"
```

### Prompt 契约

#### 输入字段

```python
REFLECT_PROMPT_INPUT_FIELDS = (
    "instruction",
    "stage",
    "dag_blueprint",
    "issues_summary",
    "min_depth",
    "min_width",
    "max_depth",
    "current_depth",
)
```

#### 输出字段

```python
REFLECT_PROMPT_OUTPUT_FIELDS = (
    '{"decision": "...", "reason": "...", "missing_operators": [...], "shape_risks": [...], "structural_warnings": [...]}',
)
```

### 测试覆盖

#### test_reflect_agent_returns_structured_finish_decision

**覆盖**:
- ✅ 正常情况返回 "finish"
- ✅ reason 为非空字符串
- ✅ reflection_history 正确记录

#### test_reflect_agent_requests_replan_when_execution_gaps_exist

**覆盖**:
- ✅ 有 gap 时返回 "need_replan"
- ✅ missing_operators 正确提取

**不覆盖**:
- ❌ "halt" 决策（空 DAG）
- ❌ "need_patch" 决策（深度不足）
- ❌ shape_risks 提取
- ❌ structural_warnings 聚合

---

## Report Agent

### 架构概览

#### 调用链路

```
WorkflowState (after reflect)
    ↓
report_agent() [src/agents/report_agent.py]
    ├─→ render_report_prompt() [渲染契约]
    ├─→ llm.render_report() [生成报告]
    └─→ return markdown
    ↓
Markdown Report (str)
```

### 输入输出契约

#### 输入

```python
state: WorkflowState,                     # 包含 dag, reflection_results
protocol: DatasetProtocol,
manifest: CompiledDagManifest,            # Bridge 编译结果
path_artifacts: Dict[str, Any],          # 路径特定产物
llm: OfflineLLM,
```

#### 输出

```python
return str  # Markdown 格式报告
```

### Report 结构

#### 通用部分

```markdown
# PHMGA Report: {dataset_name} / {graph_path}

## Summary
- Instruction: {instruction}
- DAG hash: `{dag_hash}`
- Path: {graph_path}
- Reflection decision: {decision}

## Workflow Plan
- Planned steps: {step_count}

## Review Context
- Stage: {stage}
- Current depth: {current_depth}
- Issues summary: {issues_summary}
```

#### DAG Evidence (graph_path="dag_only")

```markdown
## DAG Evidence
- Node inventory: {node_count}
- Edge inventory: {edge_count}
- Method description: {method_description}
```

#### ML Evidence (graph_path="ml")

```markdown
## ML Evidence
- Feature specs: {feature_spec_count}
- Metrics keys: {metric_keys}
- Importance keys: {importance_keys}
```

#### Torch Evidence (graph_path="torch")

```markdown
## Torch Evidence
- Build plan keys: {build_plan_keys}
- Training curves: {training_curves}
- Checkpoint keys: {checkpoint_keys}
```

### Prompt 契约

#### 输入字段

```python
REPORT_PROMPT_INPUT_FIELDS = (
    "instruction",
    "graph_path",
    "compiled_manifest",
    "path_artifacts",
    "reflection_summary",
    "review_context",
)
```

#### 输出字段

```python
REPORT_PROMPT_OUTPUT_FIELDS = ("markdown report with structured sections",)
```

### 测试覆盖

#### test_report_agent_writes_dag_only_sections

**覆盖**:
- ✅ 生成 "## DAG Evidence" 部分
- ✅ 包含 reflection decision

#### test_report_agent_writes_ml_and_torch_sections

**覆盖**:
- ✅ ml 路径生成 "## ML Evidence"
- ✅ torch 路径生成 "## Torch Evidence"

**不覆盖**:
- ❌ path_artifacts 的各种字段
- ❌ 错误情况处理
- ❌ reflection_summary 格式

---

## 对比分析

### 相似点

| 方面 | Reflect Agent | Report Agent |
|------|---------------|--------------|
| 输入 | WorkflowState + dag | WorkflowState + manifest + artifacts |
| 输出 | ReflectionResult | str (Markdown) |
| 依赖 | OfflineLLM.reflect_workflow() | OfflineLLM.render_report() |
| 状态更新 | 是 | 否 |

### 不同点

| 方面 | Reflect Agent | Report Agent |
|------|---------------|--------------|
| 目的 | 决策下一步 | 生成最终报告 |
| 输出类型 | 结构化数据 | 非结构化文本 |
| 状态修改 | 更新 state | 只读 |
| 阶段 | POST_EXECUTE | FINAL_REPORT |

---

## 总结

### Reflect Agent

**优点**:
- ✅ 清晰的决策树
- ✅ 结构化的输出（ReflectionResult）
- ✅ 提取关键问题（missing_operators, shape_risks）

**需要改进**:
- ⚠️ 测试覆盖不完整（halt, need_patch）
- ⚠️ decision 语义可能需要扩展
- ⚠️ 没有修复建议

### Report Agent

**优点**:
- ✅ 支持多种 graph_path
- ✅ 结构化的报告格式
- ✅ 整合 reflection 结果

**需要改进**:
- ⚠️ 测试覆盖简单
- ⚠️ 缺少错误处理
- ⚠️ 报告内容可能过于简单

### 优先级

| 优先级 | 改进项 | Agent | 复杂度 |
|--------|--------|-------|--------|
| 中 | 补充 reflect 测试 | Reflect | 低 |
| 中 | 改进 report 内容丰富度 | Report | 中 |
| 低 | 添加修复建议 | Reflect | 中 |

---

## 相关文件

### Reflect Agent
- [src/agents/reflect_agent.py](../../src/agents/reflect_agent.py)
- [src/prompts/reflect_prompt.py](../../src/prompts/reflect_prompt.py)
- [tests/unit/test_reflect_agent.py](../../tests/unit/test_reflect_agent.py)

### Report Agent
- [src/agents/report_agent.py](../../src/agents/report_agent.py)
- [src/prompts/report_prompt.py](../../src/prompts/report_prompt.py)
- [tests/unit/test_report_agent.py](../../tests/unit/test_report_agent.py)
