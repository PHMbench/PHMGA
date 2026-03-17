# 03 Training And Evaluation

## 本文档解决什么问题

本文档明确：

1. 三条 graph path 现在分别产出什么
2. `src/data` 与 `src/model` 的子能力在后端主链中处于什么位置
3. 哪些路径是真实可用，哪些仍然只是合同验证态
4. `dag_quality_evaluator` 的质量摘要如何进入反思与报告

## Graph-dependent artifacts

### `dag_only`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `dag_graph.md`
- `dag_artifacts.json`
- `decision_side_outputs.json`
- `method_description.md`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `dag_quality_summary.json`
- `final_report.md`

### `ml`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `decision_side_outputs.json`
- `metrics.json`
- `predictions.json`
- `importance.json`
- `similarity_artifacts.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `dag_quality_summary.json`
- `final_report.md`

### `torch`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `model_build_plan.json`
- `decision_side_outputs.json`
- `training_curves.json`
- `checkpoint.json`
- `importance.json`
- `metrics.json`
- `predictions.json`
- `similarity_artifacts.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `dag_quality_summary.json`
- `final_report.md`

## `dag_quality_evaluator` 的位置

当前 compact evaluator 位于：

`execute_agent -> dag_quality_evaluator -> reflect_agent`

它输出 `dag_quality_summary.json`，供：

- `reflect_agent` 做当前 round 的 finish / patch / replan 辅助判断
- `report_agent` 写一个简短的 `DAG Quality` section

它不是：

- 第五个主路径 agent
- 全量训练反馈系统
- bridge 的一部分

## data / model 子能力如何进入后端链

### `src/data/dataset_preparer.py`

位置：

- `bridge -> dataset_preparer -> downstream path runner`

职责：

- 从 split records 和 compiled feature plan 构建 `DatasetView`
- 固定输出 train / val / test 视图

当前事实：

- `ml / torch` 已消费 compiled subgraph，而不再只消费单链 `FeatureSpec`
- multi-parent 节点已经可以进入 compiled output feature
- output selection 当前由 config-driven `output_policy` 控制

### `src/model/shallow_ml.py`

位置：

- `ml` path 内部 baseline 层

职责：

- 对 `DatasetView` 运行 shallow baselines
- 当前支持：
  - `logistic_regression`
  - `random_forest`
  - `svm`

### `src/model/inquirer.py`

位置：

- `ml / torch` path 的 optional analysis side branch

职责：

- 根据 dataset views 构建 similarity artifacts
- 当前输出例如：
  - `split_sizes`
  - `class_centroids`
  - `class_centroid_similarity`
  - `test_to_train_mean_similarity`

它是报告证据的 side artifact，不是第五个主路径 agent。

## Proxy probe 开关

`dag_quality_evaluator` 的质量摘要分两层：

- 无监督健康度
  - 始终可算，例如 `nan_ratio` 与 `zero_variance_ratio`
- 有监督小样本 probe
  - 由 `evaluation.dag_quality.use_proxy_probe` 控制
  - 当前只返回一个紧凑的 `proxy_probe_macro_f1`

当前默认语义：

- synthetic
  - `enabled: true`
  - `use_proxy_probe: false`
- real
  - `enabled: true`
  - `use_proxy_probe: true`

如果 `use_proxy_probe: null`，运行时按 `source_mode` 自动推断以上默认值。

## 当前实现态

### `dag_only`

- 已经最接近论文叙事主链
- 当前能够稳定产出 DAG / manifest / report
- 当前 enriched 样例已经能同时包含 `EXPAND / TRANSFORM / AGGREGATE / MULTI_VARIABLE / DECISION`
- 当前也会导出 `dag_quality_summary.json`
- 但前端执行仍是 preview-level，而不是 full dataset execution

### `ml`

- 已经通过：
  - `dataset_preparer`
  - `shallow_ml`
  - `inquirer`
形成更清晰的后端分层
- 当前 richer feature pipeline 已允许 `EXPAND -> AGGREGATE` 与 `TRANSFORM -> AGGREGATE` 分支共同进入 bridge
- 当前主要作用：
  - 验证 `DAG JSON -> feature pipeline -> dataset views -> shallow baseline -> similarity artifacts` 合同
  - 在需要时给 `dag_quality_evaluator` 提供小样本 proxy probe 的最小监督证据
- 当前 `feature_pipeline.json` 已切到：
  - `execution_nodes`
  - `output_specs`
  - `output_policy`

### `torch`

- 当前 `run_torch_pipeline()` 已切到 graph-level operator PT execution，并使用最小 torch tensor runtime
- 当前 dataset views 已分成 `np` / `pt` 两条执行面；`torch` path 使用 tensor 版 dataset views
- 当前 operator-level PT backend 已优先 native 化 `stft / psd / hilbert_envelope` 等热点；`filter` 与 `kurtosis` 仍短期保留统一 bridge
- 已经补上 similarity artifacts 和 richer evidence chain
- 当前已提供最小 `GraphModule / module factory`，并可通过 phase 切到 learnable runtime
- 仍不应描述成完整 research-grade PyTorch 训练系统
- 当前 torch runtime 仍先消费 compiled execution contract，再接最小 trainable head；它不是“把 DAG 直接当 end-to-end neural architecture graph”

当前默认开发环境同时要求：

- `.venv` 中必须安装 GPU 版 `torch+cu118`
- 当前仓库的标准运行环境是仓库根目录 `.venv`，不是外部 `conda` 环境
- operator-level PT tests 默认必须跑
- 当前标准 wheel 与 CUDA 路径以 `11.8` 为准；如果 `torch.cuda.is_available()` 仍为 `False`，应视为 GPU runtime / NVML 可见性问题，而不是 wheel 版本不匹配

## path maturity matrix

| path | 当前成熟度 | 当前真实可用性 | 当前主要风险 | 推荐下一步 |
| --- | --- | --- | --- | --- |
| `dag_only` | `M2` | 可导出合法 DAG、manifest 和报告 | 仍是 preview execution | 稳住 multi-round prompts / tests |
| `ml` | `M2` | dataset views + shallow baseline + similarity 已可跑 | operator 覆盖仍窄 | 先扩一批 PHM 常见算子 |
| `torch` | `M2+` | graph-level PT execution + 最小 torch runtime 已可跑，且已提供 opt-in `GraphModule`/learnable runtime | trainer 仍最小化，planner 仍未主动使用新 runtime 能力 | 先稳住 `compiled` 基线，再系统化做 module/runtime ablation |

## 为什么不是直接把 DAG 当神经网络

当前 DAG 首先是 PHM feature / method graph，不是 end-to-end neural architecture graph。

这意味着：

- `compiled torch execution plan` 解决的是：
  - 哪些节点执行
  - topo order
  - multi-parent 输入绑定
  - 哪些节点作为最终输出
- `nn.Module` 解决的是：
  - 参数注册
  - `state_dict`
  - optimizer
  - gate / attention / `tau` 等 learnable behavior

因此未来正确结构不是二选一，而是：

`bridge -> compiled execution plan -> torch runtime -> GraphModule / module factory`

不能跳过 compiled plan 直接“按 DAG 搭网络”。原因是：

- multi-parent 的核心问题首先是编译与输出选择问题
- `decision` 节点仍然是 side-output，不属于训练张量主链
- 即使未来每个算子都可引入 gate 或 attention，runtime 也仍然需要一个稳定的 compiled topology

## `compiled subgraph` 与 `compiled output feature`

这里需要固定两层概念：

- `compiled subgraph`
  - 为 `ml / torch` path 选出的最小可执行子图
- `compiled output feature`
  - `compiled subgraph` 执行完后真正进入最终 `X` 的输出节点

在旧 single-parent 场景里，这两层曾经被 `FeatureSpec` 混成一层；当前 bridge 已开始显式区分这两层。

### enriched DAG 对比例子

当前 enriched 样例下，bridge 已开始区分：

- `compiled subgraph`
  - 包含中间 `feature` 与 `multi` 节点
  - 例如：
    - `normalize`
    - `stft`
    - `spectral_centroid`
    - `psd`
    - `band_power`
    - `cross_correlation`
    - `concatenate`
- `compiled output features`
  - 当前可由 policy 控制：
    - `terminal_only`
    - `include_intermediate_features`
  - 默认 `terminal_only` 会优先保留 terminal `feature|multi`
  - `include_intermediate_features` 则允许中间 `feature` 节点与 terminal `multi` 一起暴露

`threshold` 继续只作为 side-output，原因是：

- `multi` 节点仍然属于表征与融合的一部分
- `decision` 节点表达的是 terminal rule / evidence，不是训练特征本体

## 未来 phase 与 config 方向

以下 phase 现在已经形成明确分层，其中 `phase_3/phase_4` 已有最小实现，但默认仍不启用：

- `phase_1_fixed_compiled_runtime`
  - 目标：固定 compiled graph / output policy / dataset_preparer / `ml/torch` runner contract
  - 当前优先级最高
- `phase_2_provider_backed_llm`
  - 目标：通过 provider mode 接通 OpenRouter client，但不改变 DAG/bridge 合同
- `phase_3_module_runtime`
  - 目标：torch runtime 将 compiled nodes materialize 成 `nn.Module`
  - 当前已支持 fixed node modules 和最小 `GraphModule`
- `phase_4_learnable_control`
  - 目标：给节点加入 gate / `softmax(logits / tau)` / attention / strength control
  - 当前已支持 runtime-level gate、单父 `channel_self_attention` 与多父 `attention_fusion`

建议的未来 config 方向如下：

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

    control:
      default_mode: fixed
      tau: 1.0
      attention_heads: 1
      attention_dropout: 0.0
```

默认解释固定为：

- `phase=compiled`
  - 当前默认
- `module_runtime.enabled=false`
  - 当前默认
- `control.default_mode=fixed`
  - 当前默认；`learnable_control` 下可切到 `gated` 或 `attention`

这些键位不再只是路线图；当前仓库已经实现真实的 phase 切换，但默认主链仍保持 `compiled + fixed`。

当前已经实现的 config 控制面：

```yaml
model:
  ml:
    output_policy: terminal_only
  torch:
    phase: compiled
    output_policy: terminal_only
```

还要明确一条优先级规则：

- GraphModule、gate、attention 都建立在 fixed compiled/runtime correctness 已稳定的前提上
- OpenRouter provider path 已可选启用，但优先级仍低于 fixed graph correctness
- 当前 `offline_stub` 必须继续保留，作为 deterministic baseline 和测试基线

## 报告合同

当前 `report_agent` 仍使用 deterministic / rule-based renderer：

- `report_agent` 负责收集 graph-dependent artifacts、reflection summary 和 review context
- `OfflineLLM.render_report()` 负责把这些证据稳定地组织成 markdown

这不是 provider-backed report 的退化版本，而是当前阶段的正式默认值。这样做的原因是：

- 保证报告输出可复现
- 保证 artifact 到报告段落的映射稳定
- 避免在主链和 bridge 还在收敛时引入额外的 LLM 不确定性

当 `llm.mode=provider` 时：

- planner、param resolver、reflector、reporter 会改走 OpenRouter-backed client
- 结构化阶段仍需满足本地 schema 校验
- `offline_stub` 仍然是默认和测试基线

### `dag_only`

强调：

- 结构先验
- 节点与边的解释
- manifest 和方法说明

### `ml`

强调：

- feature pipeline
- shallow baseline 指标
- importance
- similarity artifacts

### `torch`

强调：

- `ModelBuildPlan`
- training curves
- checkpoint
- importance
- similarity artifacts

## 当前明确边界

- `graph_path` 由 Hydra root config/runtime 决定，不由 planner prompt 输入决定
- `report_agent` 只消费 artifacts，不生产 artifacts
- `execute_agent` 只允许优化 operator params，不允许改训练器参数
- `decision` 当前是 terminal side-output，可执行并进入 manifest / report，但不进入训练张量主链
- `min_depth` 继续保留，但在 reflect 中只是软约束，不再是唯一 finish 规则
