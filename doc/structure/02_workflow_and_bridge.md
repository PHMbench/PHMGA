# 02 Workflow And Bridge

## Workflow front-end

前端只保留三层：

- `states/`
- `prompts/`
- `agents/`

它们共同负责生成和修订 DAG 结构先验。

边界约束：

- `agents` 只能读写 `WorkflowState`。
- `agents` 只能通过 `dag/` 生成或修改 DAG。
- `agents` 不得直接修改 `model/`、`training/` 内部参数。

## 法定接口

前后端唯一法定接口是 `validated DAG JSON`。

正式链路：

`agents / prompts / states -> DAGTracker(NetworkX) -> DAG JSON -> bridge -> {DAG-only | ML | Torch}`

## Bridge 输出对象

- `DagArtifacts`
- `FeaturePipelinePlan`
- `ModelBuildPlan`

无论哪条路径，都必须额外落盘 `compiled_dag_manifest.json`，至少包含：

- DAG hash
- 拓扑序
- 节点清单
- `op_uid`
- backend availability
- shape inference
- path type
- 编译告警

## Graph path 选择

- `dag_only`: 直接输出结构图、JSON、节点边清单和方法说明。
- `ml`: 编译为特征流水线并交给轻量 ML 终端。
- `torch`: 编译为可训练模型构建计划并进入训练与解释阶段。
