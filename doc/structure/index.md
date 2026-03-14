# PHMGA Structure Index

主链：

`problem -> protocol -> workflow -> dag -> operators -> bridge -> model -> training -> evaluation -> rebuild`

## 当前实现态视角

- `problem/protocol` 已落到 `config/` + `src/data/`
- `workflow/dag/operators/bridge` 已落到 `src/states`、`src/agents`、`src/dag`、`src/operators`、`src/bridge`
- `training/evaluation/report` 已落到 `src/training`、`src/evaluation` 与 `scripts/run_case.py`
- 当前文档是解释现代码骨架的权威说明，不是等待未来实现的占位页

三条 graph 路径：

- `DAG-only`
- `DAG + ML / AutoML`
- `DAG + Torch trainable model`

文档导航：

- `00_problem_and_protocol.md`: 论文问题、canonical metadata、split 和 window 语义。
- `01_dag_and_operators.md`: DAG IR、JSON 校验链路、一魂三体算子契约。
- `02_workflow_and_bridge.md`: agents/prompts/states 边界，以及 DAG JSON 到不同后端的桥接。
- `03_training_and_evaluation.md`: graph-dependent artifacts、训练、评估和报告导出。
- `05_missing_assets_and_roadmap.md`: 当前尚未补齐的 prompt/agent/LLM/path 合同，以及 recommended default。
- `04_rebuild_checklist.md`: 重建顺序、目录骨架、验收标准。
- `del/`: 删除政策和旧分支清单。

当前审阅辅助文档：

- `del/02_agent_review_findings.md`: 基于 `feature-NSNet` 的算子与 prompts 资产，对当前 `journal_thesis` 的问题清单、根因、风险等级与修复优先级。
