# PHMGA Structure Index

主链：

`problem -> protocol -> workflow -> dag -> operators -> bridge -> model -> training -> evaluation -> rebuild`

三条 graph 路径：

- `DAG-only`
- `DAG + ML / AutoML`
- `DAG + Torch trainable model`

文档导航：

- `00_problem_and_protocol.md`: 论文问题、canonical metadata、split 和 window 语义。
- `01_dag_and_operators.md`: DAG IR、JSON 校验链路、一魂三体算子契约。
- `02_workflow_and_bridge.md`: agents/prompts/states 边界，以及 DAG JSON 到不同后端的桥接。
- `03_training_and_evaluation.md`: graph-dependent artifacts、训练、评估和报告导出。
- `04_rebuild_checklist.md`: 重建顺序、目录骨架、验收标准。
- `del/`: 删除政策和旧分支清单。
