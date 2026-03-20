# Torch Graph Module And Phases

## Current State

当前 graph-level `torch` path 已经具备这些事实：

- operator-level `forward_pt` 已可调用
- `run_torch_pipeline()` 已能消费 graph-level PT execution
- 当前 trainer 仍是最小线性头
- `decision` 仍然只做 terminal side-output
- `ml / torch` compiled plan 已切到：
  - `execution_nodes`
  - `output_specs`
  - `output_policy`
- 当前 output policy 已支持：
  - `terminal_only`
  - `include_intermediate_features`
- 当前 `torch` path 已新增最小 `GraphModule / module factory`
- 当前 `learnable_control` 已支持：
  - gate
  - 单父 `channel_self_attention`
  - 多父 `attention_fusion`

当前主缺口已经收敛为：

- trainer 仍是最小线性头
- planner / execute agent 仍未主动生成和补参这批新 runtime 能力
- WaveFilters family 仍未进入系统化论文消融与主表

## Why Compiled Plan First

### 核心判断

multi-parent 的首要问题是编译问题，不是纯 `nn.Module` 问题。

原因是当前需要先回答：

- 哪些节点属于当前 graph path 的最小可执行子图
- 哪些节点是最终输出
- 哪些节点只是中间计算
- `decision` 节点为什么不进入训练张量主链

这些问题都属于 bridge/path compilation 的职责，不属于 `plan_agent`，也不属于单个 node module 自身。

### `compiled subgraph`

`compiled subgraph` 是为某个 graph path 选出的最小可执行子图。

它回答：

- 为了得到后端真正要消费的结果，哪些节点必须执行

### `compiled output feature`

`compiled output feature` 是 `compiled subgraph` 执行完后真正进入最终 `X` 的输出节点。

它回答：

- 最终哪些节点结果会组成特征矩阵

### enriched DAG example

以当前 enriched DAG 为例：

- 旧 single-parent `FeatureSpec` 只能自然表达：
  - `spectral_centroid`
  - `kurtosis`
  - `band_power`
- 但 richer 方法链里真正有意义的最终输出可能是：
  - `cross_correlation`
  - `concatenate`

因此：

- `compiled subgraph`
  - 需要包含中间 `feature` 与 `multi` 节点
- `compiled output feature`
  - 当前目标允许来自 `feature | multi`
- `decision.threshold`
  - 继续只做 side-output

### 为什么不能直接“按 DAG 搭网络”

当前 DAG 首先是 PHM feature/method graph，而不是 end-to-end neural architecture graph。

所以：

- `compiled execution plan`
  - 负责节点执行顺序、输入绑定、输出选择
- `nn.Module`
  - 负责参数注册、`state_dict`、optimizer、gate、attention、`tau`

未来即使每个算子都引入 learnable gate，也仍然需要先有稳定的 compiled topology。

## Future GraphModule Direction

未来 torch runtime 的合理方向不是跳过 compiled plan，而是：

`bridge -> compiled execution plan -> GraphModule / module factory -> trainer`

这里建议把 runtime 分成两层：

### 1. compiled execution layer

负责：

- topo-order execution
- node cache
- multi-parent 输入装配
- output node 选择

### 2. module runtime layer

负责把 compiled nodes materialize 成运行时对象：

- `fixed module`
  - 当前固定 operator 的 module wrapper
- `gated module`
  - 带 learnable gate 的 node wrapper
- `attention fusion module`
  - 适合 `multi` 节点的 learnable fusion wrapper

### 未来可学习控制

如果后续对每个算子施加 learnable control，更合理的是分级演进：

- `fixed`
  - 完全固定 operator
- `tunable_params`
  - 节点参数可调，但结构固定
- `gated`
  - 节点结果受 learnable gate 控制
- `attention_fusion`
  - 多输入节点用 attention 融合

`decision` 节点仍不进入训练主链；它最多是 auxiliary evidence branch。

## Future TRANSFORM Family: WaveFilters

以下模块已经进入统一 operator 系统，但当前默认仍不要求 planner 主动使用：

- `WaveFilters`
- `RickerWaveletFilter`
- `ChirpletWaveletFilter`
- `LaplaceWaveletFilter`
- `MorletWaveletFilter`

这里的关键判断是：

- 它们虽然来源于 `nn.Module` 实现
- 但在论文版仓库中不应直接以“纯 torch module 家族”进入
- 更合理的方式是先统一成 operator contract，再在 `GraphModule` 阶段 materialize 成 runtime node

因此文档上先固定它们的归位：

- 属于 `TRANSFORM` family
- 统一归入 `src/operators/transform_ops.py`
- 不单独开 `wavefilters` config subtree

### 为什么它们仍应覆盖 `np / pt / sym`

这组模块未来如果进入论文版仓库，仍要服从统一 operator contract：

- `np`
  - 固定数值实现或近似实现
- `pt`
  - 当前最自然的主实现
- `sym`
  - planner / report / manifest 用的符号表达

也就是说：

- 它们不是“先做纯 torch module，以后再考虑 operator”
- 而是“先统一为 operator，再在 module runtime 阶段增强其 learnable behavior”

### 参数策略

这组 future `TRANSFORM` nodes 的参数也不单独开 config 面。

统一参数来源应继续保持为：

- `OperatorSpec.param_defaults`
- `StepPlan.params`
- `execute_agent` 基于 `signal_context / parent summary / operator schema` 的补全

不新增：

- `wavefilters.family`
- `wavefilters.sigma`
- `wavefilters.f_c`
- `wavefilters.f_b`

等独立 config 树。

## Phase-by-Phase Evolution

### `phase_1_compiled`

目标：

- 已完成 multi-parent compiled support
- 已让 `feature | multi` 成为合法 output node
- 当前继续使用 minimal trainer

当前默认：

- `phase=compiled`

### `phase_2_provider_backed_llm`

目标：

- 接通 OpenRouter client
- 保持 `offline_stub` 作为 deterministic fallback
- 不改变 DAG / bridge compiled contract

### `phase_3_module_runtime`

目标：

- torch runtime 将 compiled nodes materialize 成 `nn.Module`
- 支持 fixed node modules
- 引入最小 `GraphModule`

当前状态：

- 已实现
- 默认仍不启用

此时系统结构变成：

- bridge 仍负责 compiled plan
- runtime 负责 module 实例化

### `phase_4_learnable_control`

目标：

- 在 node runtime 中加入 gate
- 支持 `softmax(logits / tau)`
- 支持 attention-style fusion
- 支持控制 learned structure strength

当前状态：

- 已实现最小 runtime-level control
- 当前 attention 语义固定为：
  - 单父：`channel_self_attention`
  - 多父：`attention_fusion`
- 默认仍不启用

建议未来 config 方向：

```yaml
llm:
  provider: openrouter
  mode: offline_stub | provider

model:
  torch:
    phase: compiled
    device: auto

    module_runtime:
      enabled: false

    gate:
      enabled: false
      mode: softmax_tau
      tau: 1.0

    attention:
      enabled: false
      mode: node_fusion
```

默认解释：

- `phase=compiled`
  - 当前默认
- `module_runtime.enabled=false`
  - 当前默认
- `gate.enabled=false`
  - 当前默认
- `attention.enabled=false`
  - 当前默认

这些键位当前只用于冻结演进方向，不表示仓库已经实现真实 phase 切换。

当前优先级规则也要写死：

1. 先把 fixed compiled/runtime graph 跑稳
2. 再接 provider-backed OpenRouter
3. 再做 `GraphModule`
4. 最后才引入 gate / attention / learnable control

## Future Interfaces To Freeze

当前建议未来 bridge/runtime 围绕以下接口演进：

```python
class CompiledExecutionNode:
    node_id: str
    op_uid: str
    kind: str
    parents: list[str]
    input_bindings: dict[str, str]
    params: dict[str, object]
    channel_index: int | None

class CompiledOutputSpec:
    output_node_id: str
    output_kind: str
```

未来 torch runtime 的最小目标不是“把 DAG 直接当网络”，而是：

- 先有稳定的 compiled execution plan
- 再在此基础上引入 `GraphModule`
- 最后才引入 gate / attention / learnable control
