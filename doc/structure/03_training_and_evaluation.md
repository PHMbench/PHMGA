# 03 Training And Evaluation

## 本文档解决什么问题

本文档明确：

1. 三条 graph path 现在分别产出什么
2. `src/data` 与 `src/model` 的子能力在后端主链中处于什么位置
3. 哪些路径是真实可用，哪些仍然只是合同验证态

## Graph-dependent artifacts

### `dag_only`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `dag_graph.md`
- `dag_artifacts.json`
- `method_description.md`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `final_report.md`

### `ml`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `feature_pipeline.json`
- `metrics.json`
- `predictions.json`
- `importance.json`
- `similarity_artifacts.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `final_report.md`

### `torch`

当前输出：

- `dag.json`
- `compiled_dag_manifest.json`
- `model_build_plan.json`
- `training_curves.json`
- `checkpoint.json`
- `importance.json`
- `metrics.json`
- `predictions.json`
- `similarity_artifacts.json`
- `resolved_splits.json`
- `resolved_dataset_manifest.json`
- `workflow_state.json`
- `final_report.md`

## data / model 子能力如何进入后端链

### `src/data/dataset_preparer.py`

位置：

- `bridge -> dataset_preparer -> downstream path runner`

职责：

- 从 split records 和 compiled feature specs 构建 `DatasetView`
- 固定输出 train / val / test 视图

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

## 当前实现态

### `dag_only`

- 已经最接近论文叙事主链
- 当前能够稳定产出 DAG / manifest / report
- 但前端执行仍是 preview-level，而不是 full dataset execution

### `ml`

- 已经通过：
  - `dataset_preparer`
  - `shallow_ml`
  - `inquirer`
 形成更清晰的后端分层
- 当前主要作用：
  - 验证 `DAG JSON -> feature pipeline -> dataset views -> shallow baseline -> similarity artifacts` 合同

### `torch`

- 当前 `run_torch_pipeline()` 仍是 NumPy fallback
- 已经补上 similarity artifacts 和 richer evidence chain
- 仍不应描述成正式 PyTorch 训练系统

## path maturity matrix

| path | 当前成熟度 | 当前真实可用性 | 当前主要风险 | 推荐下一步 |
| --- | --- | --- | --- | --- |
| `dag_only` | `M2` | 可导出合法 DAG、manifest 和报告 | 仍是 preview execution | 稳住 multi-round prompts / tests |
| `ml` | `M2` | dataset views + shallow baseline + similarity 已可跑 | operator 覆盖仍窄 | 先扩一批 PHM 常见算子 |
| `torch` | `M1` | artifact contract 与 fallback path 可验证 | 仍是 NumPy fallback | 等 operator / bridge 稳定后再升级真正 torch |

## 报告合同

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

- `graph_path` 由 config/runtime 决定，不由 planner prompt 输入决定
- `report_agent` 只消费 artifacts，不生产 artifacts
- `execute_agent` 只允许优化 operator params，不允许改训练器参数
- `decision` 仍未进入正式可执行链
