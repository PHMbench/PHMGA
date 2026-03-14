# 03 Training And Evaluation

## 本文档解决什么问题

本文档明确：

1. 三条 graph path 现在分别产出什么
2. 哪些路径是真实可用，哪些仍是占位实现
3. 当前 `OfflineLLM` 与离线训练器处于什么成熟度阶段

## Graph-dependent artifacts

### `dag_only`

当前应输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `dag_graph.md`
- `dag_artifacts.json`
- `method_description.md`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `final_report.md`

### `ml`

当前应输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `metrics.json`
- `predictions.json`
- `importance.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `final_report.md`

### `torch`

当前应输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `model_build_plan.json`
- `training_curves.json`
- `checkpoint.json`
- `importance.json`
- `metrics.json`
- `predictions.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `final_report.md`

## 当前实现态

### `ml`

- 使用 `LogisticRegression` 作为最小 baseline
- 通过 bridge 编译后的 `FeaturePipelinePlan` 构建特征矩阵
- 当前主要作用是验证 `DAG JSON -> feature plan -> metrics` 的合同

### `torch`

- 当前 `run_torch_pipeline()` 仍是 NumPy fallback
- 其目标是验证：
  - `ModelBuildPlan`
  - `training_curves`
  - `checkpoint`
  - `importance`
  - `metrics`
- 它不是正式 PyTorch 训练器，也不应在论文文档中表述成“已完成 torch 训练系统”

### `dag_only`

- 目前是三条路径里最接近论文叙事的路径
- 但其前端 DAG 仍是固定生成，不是真正的 LLM plan-driven generation

## path maturity matrix

| path | 当前成熟度 | 当前真实可用性 | 当前主要风险 | 推荐修复顺序 |
| --- | --- | --- | --- | --- |
| `dag_only` | `M2` | bridge 与 artifact 可用 | `execute_agent` 仍绕过 plan | `P0` |
| `ml` | `M1` | baseline 可跑 | feature plan 仍过于线性，算子覆盖太窄 | `P1` |
| `torch` | `M0` | 仅 artifact contract 可验证 | 训练器是 NumPy fallback，不是正式 torch | `P2` |

这里的成熟度含义：

- `M2`: 合同和 artifact 基本可用，但前端/语义仍需补强
- `M1`: 最小演示可跑，但研究能力不足
- `M0`: 占位实现，只能用于验证接口，不应过度宣称

## 训练参数三分法

- 结构参数：由 `StepPlan + DAG JSON + bridge` 决定
- 连续可微参数：由 `ModelBuildPlan` 和真实 torch-side 训练器决定
- 数据窗口超参数：由 canonical protocol 决定

当前问题是：前两者的边界还不够干净，因为 front-end 刚从 placeholder 迁移到结构化 `StepPlan`。

## 报告合同

### `dag_only` 报告

必须强调：

- 结构先验是什么
- 每个节点为什么存在
- 节点与边如何对应方法步骤
- 为什么该结构对当前数据集与 graph path 合法

### `ml` 报告

必须强调：

- 编译出的特征流水线
- baseline 指标
- feature/operator importance
- 错误预测与限制

### `torch` 报告

必须强调：

- `ModelBuildPlan`
- 训练曲线
- checkpoint 与初始化
- 解释性结果

当前 `report_agent` / `build_final_report()` 还没有充分把这三种报告风格区分开，这属于后续代码重构项。

## LLM backend roadmap

当前状态：

- `src/llm/client.py` 中的 `OfflineLLM`
- `mode = offline_stub`

这是合同验证态，不是最终论文态。

切换到 provider-backed OpenRouter 之前，必须先冻结：

1. `plan_prompt / execute_prompt / reflect_prompt / report_prompt`
2. `StepPlan / ReflectionResult / DagJson`
3. `compiled_dag_manifest.json` 与 graph-dependent artifact contract
4. 至少一个真实生成案例的测试

否则切到真实 provider 只会把当前 prompt/agent 缺口放大，而不会提升论文质量。

## baseline、ablation 与证据链

当前最小仓库应优先保证：

- baseline 存在
- artifact 稳定落盘
- 结构证据、训练证据、报告证据彼此可对照

当前最关键的证据链是：

`StepPlan -> DAG JSON -> compiled_dag_manifest.json -> path artifacts -> final_report.md`

而不是先堆更多模型壳。

## smoke 与真实数据

- 仓库默认 smoke 继续用 synthetic run configs
- 真实 `RM_101_THU_GEARBOX` 与 `RM_017_Ottawa19` 当前主要用于：
  - protocol 构建
  - preflight
  - `dag_only` 集成

在前端 prompt/execute 合同稳定之前，不应把真实 `ml` / `torch` 路径包装成“已完成论文主实验”。

## 当前前端与报告默认

- `graph_path` 由 config/runtime 选择，不由 `plan_agent` prompt 输入决定
- `plan_agent` 输出当前运行时 `StepPlan`
- `execute_agent` 把代表性结果写回 state
- `report_agent` 以 graph-dependent artifacts 为主输入，以 review context 为辅输入
