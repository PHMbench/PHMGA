# 03 Training And Evaluation

## Graph-dependent artifacts

- `DAG-only`: `dag.json`、结构图、节点边清单、方法说明、`final_report.md`
- `ML`: `dag.json`、`compiled_dag_manifest.json`、`feature_pipeline.json`、`metrics.json`、`predictions.json`、`final_report.md`
- `Torch`: `dag.json`、`compiled_dag_manifest.json`、`model_build_plan.json`、`training_curves.json`、`checkpoint.json`、`importance.json`、`final_report.md`

## 训练参数三分法

- 结构参数：由 DAG 和 bridge 决定。
- 连续可微参数：由模型训练器决定。
- 数据窗口超参数：由 canonical protocol 决定。

## 当前实现态

- `ml` 路径当前使用 `LogisticRegression` 作为最小可运行基线。
- `torch` 路径当前通过 `run_torch_pipeline()` 输出 trainable path 所需 artifact，但训练器实现是 NumPy fallback，不是完整 PyTorch 后端。
- 两条路径都共享 `build_feature_matrix()` 生成的 split-specific 特征矩阵。
- 仓库默认 smoke 使用 synthetic run configs；真实 `RM_101_THU_GEARBOX` 与 `RM_017_Ottawa19` 目前只要求 protocol / `dag_only` 集成可运行。

## 指标与报告

- 主指标：分类准确率与 macro F1。
- 次指标：每类预测、feature/operator importance、编译告警摘要。
- 最终报告必须先给结论，再给结构证据、训练证据和 graph-dependent artifact 索引。

## 当前 artifact 文件名

- `ml`: `feature_pipeline.json`、`metrics.json`、`predictions.json`、`importance.json`
- `torch`: `model_build_plan.json`、`training_curves.json`、`checkpoint.json`、`importance.json`、`metrics.json`
- 最终报告统一为 `final_report.md`
- 所有路径都会额外落盘 `resolved_splits.json` 和 `resolved_dataset_manifest.json`

## 论文图表来源

- DAG 图：来自 `dag_only` 或任一路径共享的结构导出。
- 训练曲线：来自 `torch` 路径。
- 特征/算子重要性：来自 `ml` 与 `torch` 路径的解释性输出。
