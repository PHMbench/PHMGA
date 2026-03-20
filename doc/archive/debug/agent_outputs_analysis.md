# 四个 Agent 的输出维度和格式分析

## 测试执行结果

✅ 所有测试通过：
- `test_plan_agent.py`: 2 passed in 0.61s
- `test_execute_agent.py`: 2 passed in 0.41s
- `test_reflect_agent.py`: 2 passed in 0.60s
- `test_report_agent.py`: 2 passed in 0.41s

---

## 1. Plan Agent 输出

### State 字段修改
| 字段 | 类型 | 说明 |
|------|------|------|
| `signal_context` | `SignalContext` | 首次运行时创建 |
| `step_plan` | `StepPlan` | 核心输出 |
| `status` | `str` | 变为 `"planned"` |

### StepPlan 结构 (核心输出)
```python
{
  "plan": [
    {
      "parent": str,        # 父节点ID，可能逗号分隔多个
      "op_name": str,       # 算子名称 (如 "feature.mean")
      "params": dict        # 算子参数
    }
  ]
}
```

### SignalContext 结构
```python
{
  "dataset_name": str,                 # 数据集名称 (如 "RM_101_THU_GEARBOX")
  "channel_count": int,                # 通道数 (如 8)
  "window_shape": List[int],           # 窗口形状 (如 [8, 4096])
  "sampling_rate": int,                # 采样率 (如 10240)
  "source_mode": str,                  # 来源模式 ("real" 或 "synthetic")
  "root_node_ids": List[str],          # 根节点ID (["ch1", "ch2", ...])
  "representative_sample_id": str,     # 代表性样本ID
  "available_splits": List[str]        # 可用分割 (["train", "val", "test"])
}
```

### 契约验证
- ✅ `test_plan_agent_outputs_nvta_style_step_plan`: 验证输出格式为 `{"plan": [...]}`
- ✅ 每个步骤包含 `{"parent", "op_name", "params"}`
- ✅ 第一个步骤的 parent 以 "ch" 开头（通道节点）

---

## 2. Execute Agent 输出

### State 字段修改
| 字段 | 类型 | 说明 |
|------|------|------|
| `dag` | `DagJson` | 核心 DAG 输出 |
| `execution_results` | `Dict[str, np.ndarray]` | 节点ID -> 执行结果 |
| `execution_gaps` | `List[ExecutionGap]` | 执行间隙/错误 |
| `status` | `str` | 变为 `"executed"` |

### DagJson 结构 (核心输出)
```python
{
  "nodes": [
    {
      "node_id": str,
      "op_uid": str,                  # 算子唯一标识
      "name": str,                    # 算子显示名称
      "kind": str,                    # "input", "feature", "multi", "decision", "transform"
      "operator_category": str,
      "params": dict,                 # 实际使用的参数
      "parents": List[str],           # 父节点ID列表
      "in_shape": List[int],          # 输入形状
      "out_shape": List[int],         # 输出形状
      "backend_availability": List[str],  # ["np", "pt", "sym"]
      "execution_role": str,          # "fixed" 或 "planned"
      "legal_paths": List[str],       # ["dag_only", "ml", "torch"]
      "input_bindings": dict,         # multi 算子的参数绑定 {"arg0": "ch1", ...}
      "plan_step_ref": str,           # 引用步骤 (如 "step_01")
      "rationale": str                # 生成理由
    }
  ],
  "edges": []
}
```

### ExecutionGap 结构
```python
{
  "step_index": int,
  "parent": str,
  "op_name": str,
  "message": str,                     # 错误信息
  "recoverable": bool                 # 是否可恢复
}
```

### 契约验证
- ✅ `test_execute_agent_materializes_results_and_multi_node`: 验证 multi 节点正确生成
- ✅ `test_execute_agent_records_unknown_operator_gap`: 验证未知算子正确记录为 gap

### 关键行为
- **多父节点支持**: `parent` 字段支持逗号分隔 (如 `"ch1,ch2"`)
- **Multi 算子**: 当 op_uid 以 `multi.` 开头时，执行多输入版本
- **Decision 算子**: 作为 auxiliary terminal，不实际执行，记录为 recoverable gap

---

## 3. Reflect Agent 输出

### State 字段修改
| 字段 | 类型 | 说明 |
|------|------|------|
| `reflection_results` | `List[ReflectionResult]` | 反思结果列表 |
| `reflection_history` | `List[str]` | 反思历史文本 |
| `status` | `str` | 变为 `"reflected"` |

### ReflectionResult 结构 (核心输出)
```python
{
  "decision": str,                    # "finish", "need_patch", "need_replan", "halt"
  "reason": str,                      # 决策理由
  "missing_operators": List[str],     # 缺失算子列表
  "shape_risks": List[str],           # 形状风险警告
  "structural_warnings": List[str]    # 结构警告
}
```

### 契约验证
- ✅ `test_reflect_agent_returns_structured_finish_decision`: 验证正常完成返回 "finish"
- ✅ `test_reflect_agent_requests_replan_when_execution_gaps_exist`: 验证有 gap 时返回 "need_replan"

### 决策逻辑
- **finish**: 无执行间隙，满足最小深度/宽度
- **need_replan**: 存在不可恢复的执行间隙
- **need_patch**: 存在可恢复问题
- **halt**: 严重错误

---

## 4. Report Agent 输出

### 返回值
| 类型 | 说明 |
|------|------|
| `str` | Markdown 格式的最终报告 |

### State 字段修改
| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | `str` | 变为 `"reported"` |

### 报告结构
```markdown
## DAG Evidence
...
## ML Evidence (仅 graph_path=ml)
...
## Torch Evidence (仅 graph_path=torch)
...
Reflection decision: {decision}
```

### 契约验证
- ✅ `test_report_agent_writes_dag_only_sections`: 验证 dag_only 路径生成 DAG Evidence
- ✅ `test_report_agent_writes_ml_and_torch_sections`: 验证 ml/torch 路径生成对应章节

### 输入依赖
- `manifest: CompiledDagManifest` - Bridge 编译结果
- `path_artifacts: dict` - 路径特定产物 (metrics, feature_pipeline, training_curves 等)

---

## 实际输出示例 (rm101_synth_ml.yaml)

### Plan Agent
```
signal_context.channel_count: 2
signal_context.window_shape: [2, 256]
signal_context.root_node_ids: ['ch1', 'ch2']
step_plan.plan 长度: 11
第一个步骤: {'parent': 'ch1', 'op_name': 'normalize', 'params': {'eps': 1e-06}}
```

### Execute Agent
```
dag.nodes 数量: 13
execution_results 键数量: 13
execution_gaps 数量: 0

Multi 节点示例:
  node_id: concatenate_11_rms_07_fft_03_normalize_01_ch1__rms_10_fft_04_normalize_02_ch2
  op_uid: multi.concatenate
  parents: ['rms_07_fft_03_normalize_01_ch1', 'rms_10_fft_04_normalize_02_ch2']
  in_shape: [2]
  out_shape: [2]
  input_bindings: {'arg0': 'rms_07_fft_03_normalize_01_ch1', 'arg1': 'rms_10_fft_04_normalize_02_ch2'}
```

### Reflect Agent
```
decision: finish
reason: The DAG is structurally valid and satisfies the current planning target.
missing_operators: []
shape_risks: []
structural_warnings: []
```

### Report Agent
返回 Markdown 格式报告，包含:
- ## DAG Evidence
- ## ML Evidence (graph_path=ml 时)
- Reflection decision: finish

## 工作流数据流

```
SignalContext (channel_count, window_shape, root_node_ids)
    ↓
StepPlan (NVTA风格: [{"parent", "op_name", "params"}])
    ↓
DagJson + execution_results (Dict[str, np.ndarray]) + execution_gaps
    ↓
ReflectionResult (decision + reason)
    ↓
Markdown Report (DAG/ML/Torch Evidence + Reflection)
```

## 未完成的功能点（按优先级）

### 高优先级
1. **Decision 合同落地**: Decision 仍是 auxiliary terminal，不是 runnable inner-loop op
2. **Provider-backed LLM**: OpenRouter 未接进真实主路径 (plan/reflect/report)

### 中优先级
3. **Rich DAG 编译器**: Bridge 对 multi-parent lineage 是最小支持，不是完整编译器

---

## 相关文件

### Agents
- [src/agents/plan_agent.py](../src/agents/plan_agent.py)
- [src/agents/execute_agent.py](../src/agents/execute_agent.py)
- [src/agents/reflect_agent.py](../src/agents/reflect_agent.py)
- [src/agents/report_agent.py](../src/agents/report_agent.py)

### State & Contracts
- [src/states/workflow.py](../src/states/workflow.py)
- [doc/structure/02_workflow_and_bridge.md](../doc/structure/02_workflow_and_bridge.md)

### Tests
- [tests/unit/test_plan_agent.py](../tests/unit/test_plan_agent.py)
- [tests/unit/test_execute_agent.py](../tests/unit/test_execute_agent.py)
- [tests/unit/test_reflect_agent.py](../tests/unit/test_reflect_agent.py)
- [tests/unit/test_report_agent.py](../tests/unit/test_report_agent.py)
