# 01 DAG And Operators

## 本文档解决什么问题

本文档冻结两类合同：

1. DAG JSON 需要表达什么，才能成为论文版前后端唯一法定接口
2. operator schema 需要表达什么，才能同时服务 planner、executor、bridge 和 report

当前 operator schema 的描述层明确参考了 `/home/user/LQ/C_Agent/PHMGA/src/tools/readme.md`，但只吸收其分类和 shape 语义，不回流旧 `src/tools` 平台壳。

## 五类 operator schema 分类

### `EXPAND`

语义：增加轴、拆分轴、把一维时序提升到时频或局部片段表示。

典型 shape：

- `(B, L, C) -> (B, N, P, C)`：`patch`
- `(B, L, C) -> (B, F, T, C)`：`stft`
- `(B, L, C) -> (B, S, L, C)`：`wavelet_transform`

### `TRANSFORM`

语义：保持“仍可继续做信号处理或特征提取”的张量语义，但改变域、尺度、幅值或长度。

典型 shape：

- `(B, L, C) -> (B, L, C)`：`normalize`, `filter`, `hilbert_envelope`
- `(B, L, C) -> (B, F, C)`：`fft`, `psd`

### `AGGREGATE`

语义：沿时间、频率或展开轴做汇聚，把信号压缩成 feature vector 或结构化 feature。

典型 shape：

- `(B, L, C) -> (B, C')`
- `(B, F, C) -> (B, C')`

### `MULTI_VARIABLE`

语义：节点需要多个父节点，不再是单链变换。

典型 shape：

- 多个 `(B, C') -> (B, C_new)`：`concatenate`
- 两个 `(B, L, C) -> (B, L, C)`：`subtract`

### `DECISION`

语义：输出不再是继续进入训练内环的信号张量，而是规则结果、峰值表、分数或文本说明。

典型 shape：

- `(L,) -> dict`
- `(B, C') -> dict`

当前默认：`DECISION` 节点是一等 DAG schema 节点，但仍只作为 auxiliary terminal / report evidence，不进入 `ml / torch` 内环训练张量。

## DAG JSON 合同

当前 `DagNode` / `DagJson` 已覆盖：

- `node_id`
- `op_uid`
- `name`
- `kind`
- `operator_category`
- `params`
- `parents`
- `in_shape`
- `out_shape`
- `backend_availability`
- `execution_role`
- `legal_paths`
- `input_bindings`
- `plan_step_ref`
- `rationale`

当前校验链路是：

`DAGTracker(NetworkX) -> DAG JSON -> validated DAG JSON -> bridge`

这里必须坚持三点：

- `NetworkX` 只用于生成期
- 只有 `validated DAG JSON` 能进入 bridge
- bridge 不读取 workflow state，只读取 DAG JSON

## 当前 operator schema 合同

当前 `OperatorSpec` 已冻结这些字段：

- `op_uid`
- `name`
- `schema_category`
- `description`
- `param_schema`
- `param_defaults`
- `param_docs`
- `input_shape_rule`
- `output_shape_rule`
- `backend_availability`
- `execution_role`
- `legal_paths`
- `planning_notes`
- `llm_tunable_params`

这些字段分别服务：

- `schema_category`
  - planner 和 report 用来讲清方法类别
- `description`
  - prompt 和报告用来表达算子语义
- `param_docs`
  - execute prompt / param resolution 用来解释参数意义
- `param_defaults`
  - executor 的第三优先级补参来源
- `legal_paths`
  - 防止路径级越权使用
- `planning_notes`
  - 给 planner 一个简短但稳定的 PHM 语义提示
- `llm_tunable_params`
  - 限制 LLM 只优化声明过的 operator params

## 当前已实现算子与 richer metadata

当前闭环只实现了最小一批算子，但它们已经带上 richer metadata：

- `signal.normalize`
- `signal.fft_mag`
- `feature.mean`
- `feature.std`
- `feature.rms`
- `multi.concatenate`

当前 role / path 约束如下：

| op_uid | schema_category | execution_role | legal_paths | 说明 |
| --- | --- | --- | --- | --- |
| `signal.normalize` | `TRANSFORM` | `fixed` | `dag_only, ml, torch` | 时域归一化 |
| `signal.fft_mag` | `TRANSFORM` | `fixed` | `dag_only, ml, torch` | 频域变换 |
| `feature.mean` | `AGGREGATE` | `outer_only` | `dag_only, ml` | 标量均值特征 |
| `feature.std` | `AGGREGATE` | `outer_only` | `dag_only, ml` | 标量标准差特征 |
| `feature.rms` | `AGGREGATE` | `outer_only` | `dag_only, ml, torch` | RMS 特征 |
| `multi.concatenate` | `MULTI_VARIABLE` | `proxy` | `dag_only, ml, torch` | 最小可运行 multi-input 融合 |

## 当前 planner / executor 如何使用 operator schema

### planner

`plan_prompt` 读取 `OperatorCatalog.summary()` 的 rich summary。当前至少可见：

- `op_uid`
- `op_name`
- `name`
- `schema_category`
- `description`
- `input_shape_rule`
- `output_shape_rule`
- `legal_paths`
- `execution_role`
- `llm_tunable_params`
- `planning_notes`

### executor

`execute_agent` 补参顺序固定为：

1. `StepPlan.params`
2. state / signal-context derived values
3. `OperatorSpec.param_defaults`
4. LLM 对 `llm_tunable_params` 做补全或优化

也就是说，LLM 只允许触碰 operator params，不允许越权触碰 training/model hyperparameters。

## 第一批待补的核心算子

本轮不扩完整五大类集合，但 roadmap 固定为：

- `signal.filter`
- `signal.hilbert_envelope`
- `signal.psd`
- `feature.kurtosis`
- `feature.crest_factor`
- `feature.band_power`
- `feature.spectral_centroid`
- `multi.concatenate` 的 richer bridge support

## 当前明确边界

- `DagNode.kind` 仍保持 `input / transform / feature / multi / decision`
- `DagNode.operator_category` 现在承载 schema-level category，而不再简单镜像 `kind`
- `multi.concatenate` 是当前唯一 runnable multi-input operator
- `DECISION` 仍未进入正式可执行链，只保留 schema / report / auxiliary 语义
