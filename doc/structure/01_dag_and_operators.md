# 01 DAG And Operators

## 本文档解决什么问题

本文档不再只描述“当前最小实现”，而是明确两件事：

1. DAG JSON 需要表达什么，才能成为论文版前后端法定接口
2. `feature-NSNet` 中五类算子里，哪些值得迁移到 `journal_thesis`，以及应如何标记 backend / path / role

## 来自 `feature-NSNet` 的五类算子边界

### `EXPAND`

语义：增加轴、拆分轴、把一维时序提升到时频或时序片段表示。

典型 shape 规则：

- `(B, L, C) -> (B, N, P, C)`：`patch`
- `(B, L, C) -> (B, F, T, C)`：`stft` / `spectrogram`
- `(B, L, C) -> (B, S, L, C)`：`wavelet_transform`
- `(B, L, C) -> (B, L, K, C)`：`vmd` / `emd`

### `TRANSFORM`

语义：不以“生成最终特征”为目标，而是保持信号/谱语义，只改变域、尺度、幅值或采样长度。

典型 shape 规则：

- `(B, L, C) -> (B, L, C)`：`normalize` / `filter` / `hilbert_envelope`
- `(B, L, C) -> (B, F, C)`：`fft` / `psd`
- `(B, L, C) -> (B, L_new, C)`：`resample`

注意：`TRANSFORM` 并不要求长度严格不变，重点在于它仍输出“可继续做信号处理或特征提取的张量”，而不是最终决策。

### `AGGREGATE`

语义：沿长度、频率、patch 或时频轴做汇聚，把信号压缩为特征向量或小型结构化特征。

典型 shape 规则：

- `(B, L, C) -> (B, C')`：统计特征
- `(B, F, C) -> (B, C')`：频谱统计特征
- `(B, L, C) -> (B, 3, C)`：`hjorth_parameters`
- `(B, L, C) -> (B, N, C)`：多频带 `band_power`

### `MULTI-VARIABLE`

语义：节点需要多个父节点，不再是单链变换。

典型 shape 规则：

- 两个 `(B, L, C) -> (B, L, C)`：`subtract` / `arithmetic` / `element_wise_product`
- 两个 `(B, L, C) -> (B, F, C)`：`coherence`
- 多个 `(B, C') -> (B, C_new)`：`concatenate`
- 两个 `(B, C') -> (B,)`：`distance`

### `DECISION`

语义：输出不再是“继续可训练的信号张量”，而是判断、分数、峰值表、规则结果或文本说明。

典型 shape 规则：

- `(L,) -> dict`：`find_peaks`
- `(B, C') -> dict`：`outlier_detection`
- `dict -> dict/bool/score`：`rule_based_decision` / `anomaly_scorer`

结论：`DECISION` 更适合作为 `dag_only` 报告证据或外环 artifact，而不是默认进入 `ml` / `torch` 的内环训练。

## 当前 `journal_thesis` DAG 合同已具备什么

当前 `DagNode` / `DagEdge` / `DagJson` 已覆盖：

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

当前校验链路已经成立：

`NetworkX DAG -> DAG JSON -> validated DAG JSON -> bridge`

当前 bridge 也坚持了正确边界：只有通过 `validate_dag_json()` 的 DAG 才能进入编译。

## 当前 DAG 合同仍待补强的点

当前最关键的 richer contract 已经加进 `DagNode`，但还有两类能力尚未成熟：

- `decision` 节点的 bridge/backend 语义仍是 auxiliary terminal
- multi-parent lineage 在 bridge 中仍然不是一等执行链

也就是说，DAG JSON 现在足够表达论文版前端合同，但后端编译仍然偏向最小主链。

## 当前算子系统问题

当前 `OperatorCatalog` 已有：

- `signal.normalize`
- `signal.fft_mag`
- `feature.mean`
- `feature.std`
- `feature.rms`
- `multi.concatenate`

这足够验证 bridge 合同，但不足以体现 PHM 信号处理结构先验，也不足以支撑真实 prompt planning。

## 执行角色标记原则

### `fixed`

传统信号处理算子，语义稳定，默认不训练：

- `normalize`
- `filter`
- `hilbert_envelope`
- `fft`
- `psd`
- `stft`
- `wavelet_transform`

### `outer_only`

主要服务结构解释、报告证据或 classical feature 导出，不直接变成可微可学习层：

- `mean`
- `std`
- `rms`
- `kurtosis`
- `crest_factor`
- `find_peaks`
- `rule_based_decision`

### `proxy`

在 `torch` path 中先以固定算子或外部实现出现，后续可用代理近似替换：

- `band_power`
- `spectral_centroid`
- `concatenate`
- `cross_correlation`

### `trainable`

第一批不建议直接从 `feature-NSNet` 的传统 `tools` 中拷贝“trainable 算子”。论文版应把 trainable 性主要放在：

- bridge 生成的可训练头
- learnable fusion / projection
- 后续单独定义的 torch-side explainable ops

结论：第一批从 `feature-NSNet` 迁移过来的核心算子，大多应标记为 `fixed`、`outer_only` 或 `proxy`，而不是强行改成 `trainable`。

## 算子迁移表

下表是当前推荐迁移路线。`np_backend / pt_backend / sym_backend` 用 `implemented / planned / unavailable` 表示当前建议状态，而不是承诺现代码已经实现。

| op_uid | category | input_shape_rule | output_shape_rule | np_backend | pt_backend | sym_backend | execution_role | recommended_path | planned_phase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `signal.normalize` | `TRANSFORM` | `CxT` | `CxT` | `implemented` | `planned` | `implemented` | `fixed` | `dag_only, ml, torch` | `phase0_now` |
| `signal.fft_mag` | `TRANSFORM` | `CxT` | `CxF` | `implemented` | `planned` | `implemented` | `fixed` | `dag_only, ml, torch` | `phase0_now` |
| `feature.mean` | `AGGREGATE` | `CxT/CxF` | `1` | `implemented` | `unavailable` | `implemented` | `outer_only` | `dag_only, ml` | `phase0_now` |
| `feature.std` | `AGGREGATE` | `CxT/CxF` | `1` | `implemented` | `unavailable` | `implemented` | `outer_only` | `dag_only, ml` | `phase0_now` |
| `feature.rms` | `AGGREGATE` | `CxT/CxF` | `1` | `implemented` | `unavailable` | `implemented` | `outer_only` | `dag_only, ml, torch` | `phase0_now` |
| `signal.filter` | `TRANSFORM` | `CxT` | `CxT` | `planned` | `planned` | `planned` | `fixed` | `dag_only, ml, torch` | `phase1_core` |
| `signal.hilbert_envelope` | `TRANSFORM` | `CxT` | `CxT` | `planned` | `planned` | `planned` | `fixed` | `dag_only, ml, torch` | `phase1_core` |
| `signal.psd` | `TRANSFORM` | `CxT` | `CxF` | `planned` | `planned` | `planned` | `fixed` | `dag_only, ml, torch` | `phase1_core` |
| `feature.kurtosis` | `AGGREGATE` | `CxT/CxF` | `1` | `planned` | `unavailable` | `planned` | `outer_only` | `dag_only, ml` | `phase1_core` |
| `feature.crest_factor` | `AGGREGATE` | `CxT` | `1` | `planned` | `unavailable` | `planned` | `outer_only` | `dag_only, ml` | `phase1_core` |
| `feature.band_power` | `AGGREGATE` | `CxT/CxF` | `BANDxC or C'` | `planned` | `planned` | `planned` | `proxy` | `dag_only, ml, torch` | `phase1_core` |
| `feature.spectral_centroid` | `AGGREGATE` | `CxF` | `1` | `planned` | `planned` | `planned` | `proxy` | `dag_only, ml, torch` | `phase1_core` |
| `multi.concatenate` | `MULTI_VARIABLE` | `C' + C' + ...` | `C_new` | `implemented` | `planned` | `implemented` | `proxy` | `dag_only, ml, torch` | `phase0_now` |
| `signal.stft` | `EXPAND` | `CxT` | `FxTxC` | `planned` | `planned` | `planned` | `fixed` | `dag_only, torch` | `phase2_tf` |
| `signal.wavelet_transform` | `EXPAND` | `CxT` | `SxTxC` | `planned` | `planned` | `planned` | `fixed` | `dag_only, torch` | `phase2_tf` |
| `multi.cross_correlation` | `MULTI_VARIABLE` | `CxT + CxT` | `CxT_corr` | `planned` | `planned` | `planned` | `proxy` | `dag_only, ml` | `phase2_tf` |
| `decision.find_peaks` | `DECISION` | `F or T` | `dict` | `planned` | `unavailable` | `planned` | `outer_only` | `dag_only` | `phase3_outer` |
| `decision.rule_based_decision` | `DECISION` | `dict/features` | `dict` | `planned` | `unavailable` | `planned` | `outer_only` | `dag_only, ml_report` | `phase3_outer` |

## 第一批最值得迁移的核心算子

如果只保留一批最小但论文价值最高的算子，优先级建议如下：

1. `signal.filter`
2. `signal.hilbert_envelope`
3. `signal.psd`
4. `feature.kurtosis`
5. `feature.crest_factor`
6. `feature.band_power`
7. `feature.spectral_centroid`
8. `multi.concatenate`

理由很简单：

- 这批算子最贴近 PHM 论文常见叙事
- 它们既能丰富 `dag_only` 结构解释，也能服务 `ml` 特征流水线
- 它们不会立刻把 `torch` path 复杂度推高到不可控

## 对 bridge 的直接约束

为了让 bridge 可持续扩展，未来 `compile_dag_for_path()` 必须不再假设：

- 所有 feature 节点都来自单父单链 lineage
- 所有有效节点都是 `input -> transform -> feature`

一旦引入：

- `MULTI_VARIABLE`
- `EXPAND`
- `DECISION`

bridge 就必须改成读正式节点合同，而不是从线性父链反推语义。

## 当前运行时默认

当前论文版代码已经固定以下默认：

- planner 输出 `StepPlan`
- executor 使用 `op_name -> op_uid` 映射再落盘为 `DagNode`
- `multi.concatenate` 是首个可运行的 multi-input 算子
- `decision` 节点仍只作为 schema / report / auxiliary terminal 语义
