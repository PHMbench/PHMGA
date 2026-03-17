# 01 DAG And Operators

## 本文档解决什么问题

本文档冻结两类合同：

1. DAG JSON 需要表达什么，才能成为论文版前后端唯一法定接口
2. operator schema 需要表达什么，才能同时服务 planner、executor、bridge 和 report

当前 operator schema 的描述层明确参考了 `/home/user/LQ/C_Agent/PHMGA/src/tools/readme.md` 与五类 schema 文件，但只吸收其分类、rank 语义和输入输出合同，不回流旧 `src/tools` 平台壳。

## 五类 operator schema 分类

### `EXPAND`

语义：增加轴、拆分轴、把一维时序提升到时频或局部片段表示。

典型 shape：

- `(B, L, C) -> (B, F, T, C)`：`stft`
- `(B, L, C) -> (B, P, L_patch, C)`：`patch`

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
- `(B, F, T, C) -> (B, C')`

### `MULTI_VARIABLE`

语义：节点需要多个父节点，不再是单链变换。

典型 shape：

- 多个 `(B, C') -> (B, C_new)`：`concatenate`
- 两个 `(B, L, C) -> (B, 1)`：`cross_correlation`

### `DECISION`

语义：输出不再是继续进入训练内环的信号张量，而是规则结果、分数或文本说明。

典型 shape：

- `(B, C') -> dict`
- `(B, 1) -> dict`

当前默认：`DECISION` 节点是一等 DAG schema 节点，当前已进入半执行态。它可以被执行为 terminal side-output、进入 manifest 和 report，但不进入 `ml / torch` 的训练张量主链。

## 为什么继续保留 `BaseIsomorphicOperator`

当前没有回退到旧 `PHMOperator / ExpandOp / AggregateOp` 的运行时继承树。保留 `BaseIsomorphicOperator` 的原因是：

- 同一算子语义仍需要同时服务 `np / pt / sym` 多后端
- 同一份 schema 还要同时服务 planner、executor、bridge 和 report
- `schema_category / rank_class` 已经足够表达五类语义，不需要再用多基类体系去重复编码

这里的 `isomorphic` 指的是“多后端 / 多表示共用同构合同”，不是指所有算子都有相同 rank 行为。

## DAG JSON 合同

当前 `DagNode` / `DagJson` 已覆盖：

- `node_id`
- `op_uid`
- `name`
- `kind`
- `operator_category`
- `rank_class`
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
- `op_name`
- `name`
- `schema_category`
- `rank_class`
- `description`
- `input_spec`
- `output_spec`
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

当前默认环境前提补充为：

- operator-level backend 以 `np / pt / sym` 为正式合同
- `.venv` 默认要求安装 GPU 版 `torch+cu118`
- 当前仓库默认环境就是仓库根目录 `.venv`，不是外部 `conda` 环境
- 这里的 `pt` 指算子级 `forward_pt` 可调用；当前 graph-level `torch` path 已消费单父 feature lineage 的 PT execution，但 multi-parent compiled support 仍未完成
- 当前 PT 实现优先 native 化 `signal.stft`、`signal.psd`、`signal.hilbert_envelope` 等热点；`signal.filter` 与 `feature.kurtosis` 仍短期保留统一 bridge

这些字段分别服务：

- `op_name`
  - 让 planner 使用紧凑、稳定的计划级名称，而不直接暴露 `op_uid`
- `schema_category`
  - planner 和 report 用来讲清方法类别
- `rank_class`
  - executor 用来检查 arity 与维度行为是否匹配
- `input_spec / output_spec`
  - 供 planner、executor、bridge 和校验共享同一份结构化输入输出合同
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

## 当前已实现算子

当前已经进入“五类首轮覆盖”：

- `signal.normalize`
- `signal.fft_mag`
- `signal.stft`
- `signal.patch`
- `signal.filter`
- `signal.hilbert_envelope`
- `signal.psd`
- `signal.wavefilters`
- `signal.wavelet_ricker`
- `signal.wavelet_chirplet`
- `signal.wavelet_laplace`
- `signal.wavelet_morlet`
- `feature.mean`
- `feature.std`
- `feature.rms`
- `feature.kurtosis`
- `feature.crest_factor`
- `feature.band_power`
- `feature.spectral_centroid`
- `multi.concatenate`
- `multi.cross_correlation`
- `decision.threshold`

## `TRANSFORM` family：WaveFilters 路线

以下模块现在已经进入统一 operator 系统，但当前默认仍不要求 planner 主动生成：

- `WaveFilters`
- `RickerWaveletFilter`
- `ChirpletWaveletFilter`
- `LaplaceWaveletFilter`
- `MorletWaveletFilter`

它们的定位固定为：

- 属于 `TRANSFORM` family
- 统一归入 [transform_ops.py](/home/user/LQ/B_Signal/PHMGA/src/operators/transform_ops.py)
- 不单独开 `wavefilters` 配置文件或独立 config subtree

它们继续服从当前统一 operator contract：

- `op_uid`
- `schema_category=TRANSFORM`
- `rank_class=rank_same`
- `input_spec`
- `output_spec`
- `param_schema`
- `param_defaults`
- `param_docs`
- `llm_tunable_params`

参数策略写死为：

- 参数来源只允许：
  - `param_defaults`
  - `StepPlan.params`
  - `execute_agent` 基于 `signal_context / parent summary / operator schema` 的补全
- 不新增单独 config 面去承载：
  - `wavefilters.family`
  - `wavefilters.sigma`
  - `wavefilters.f_c`
  - `wavefilters.f_b`
  - 等专用树状配置

也就是说，这组模块已经是“统一 operator contract 下的可调 TRANSFORM nodes”，而不是独立子系统。其 learnable behavior 继续由 torch runtime wrapper 叠加，而不是在 operator schema 里再造第二套体系。

当前 role / path 约束如下：

| op_uid | schema_category | rank_class | execution_role | legal_paths | 说明 |
| --- | --- | --- | --- | --- | --- |
| `signal.normalize` | `TRANSFORM` | `rank_same` | `fixed` | `dag_only, ml, torch` | 时域归一化 |
| `signal.fft_mag` | `TRANSFORM` | `rank_same` | `fixed` | `dag_only, ml, torch` | 全局谱变换 |
| `signal.stft` | `EXPAND` | `rank_up` | `fixed` | `dag_only, ml, torch` | 时频展开 |
| `signal.patch` | `EXPAND` | `rank_up` | `fixed` | `dag_only, ml, torch` | 局部 patch 展开 |
| `signal.filter` | `TRANSFORM` | `rank_same` | `fixed` | `dag_only, ml, torch` | 带限滤波 |
| `signal.hilbert_envelope` | `TRANSFORM` | `rank_same` | `fixed` | `dag_only, ml, torch` | 包络提取 |
| `signal.psd` | `TRANSFORM` | `rank_same` | `fixed` | `dag_only, ml, torch` | Welch PSD |
| `feature.mean` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml` | 均值特征 |
| `feature.std` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml` | 标准差特征 |
| `feature.rms` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml, torch` | RMS 特征 |
| `feature.kurtosis` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml, torch` | 冲击性特征 |
| `feature.crest_factor` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml, torch` | 瞬态峰值特征 |
| `feature.band_power` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml, torch` | 频带能量特征 |
| `feature.spectral_centroid` | `AGGREGATE` | `rank_down` | `outer_only` | `dag_only, ml, torch` | 谱心特征 |
| `multi.concatenate` | `MULTI_VARIABLE` | `multi_input` | `proxy` | `dag_only, ml, torch` | 多分支向量融合 |
| `multi.cross_correlation` | `MULTI_VARIABLE` | `multi_input` | `proxy` | `dag_only, ml, torch` | 跨分支相关性特征 |
| `decision.threshold` | `DECISION` | `terminal_decision` | `outer_only` | `dag_only, ml, torch` | terminal side-output 决策节点 |

## 当前 planner / executor 如何使用 operator schema

### planner

`plan_prompt` 读取 `OperatorCatalog.summary()` 的 richer summary。当前至少可见：

- `op_uid`
- `op_name`
- `name`
- `schema_category`
- `rank_class`
- `input_spec`
- `output_spec`
- `description`
- `input_shape_rule`
- `output_shape_rule`
- `legal_paths`
- `execution_role`
- `llm_tunable_params`
- `planning_notes`

### executor

`execute_agent` 先校验 schema，再执行：

1. 按 `input_spec / rank_class` 校验 arity 和最小 rank
2. 按固定顺序补参：
   - `StepPlan.params`
   - state / signal-context derived values
   - `OperatorSpec.param_defaults`
   - LLM 对 `llm_tunable_params` 做补全或优化
3. materialize 节点并写回 `execution_results`

也就是说，LLM 只允许触碰 operator params，不允许越权触碰 training/model hyperparameters。

## 当前下一阶段重点

当前第一批五类算子已经落地，下一阶段重点不再是“有没有这些算子”，而是：

- planner 是否能稳定利用 richer schema 生成更有 PHM 语义的 DAG
- bridge 是否能给 multi-parent lineage 更强的一等支持
- `decision` side-output 是否需要从当前阈值节点扩到更丰富的 terminal rule family

## 当前明确边界

- `DagNode.kind` 仍保持 `input / transform / feature / multi / decision`
- `DagNode.operator_category` 承载 schema-level category，而不再简单镜像 `kind`
- `DagNode.rank_class` 记录维度行为：`rank_up / rank_same / rank_down / multi_input / terminal_decision`
- `multi.cross_correlation` 与 `multi.concatenate` 都已可执行
- `DECISION` 当前是半执行：允许生成节点、执行 terminal side-output、进入 manifest 和 report，但不进入 `ml / torch` 的训练张量主链
