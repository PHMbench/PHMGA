# Legacy Inventory

| 旧内容 | 处理方式 | 原因 | 替代物 | 是否影响论文复现 |
| --- | --- | --- | --- | --- |
| 旧 `src/agents` 编排壳 | 删除后重写 | 旧实现将 workflow、训练和兼容层耦合在一起 | 新 `src/agents` + `WorkflowState` | 否 |
| 旧 `src/states/PHMState` | 删除后重写 | 混合了 `reference/test`、训练后端、报告和兼容字段 | 轻量 `WorkflowState` + path-specific result models | 否 |
| 旧 `src/tools` / `src/model/explainable` 双轨算子定义 | 抽取思想后重写 | 算子语义分散，无法形成统一合同 | 新 `OperatorCatalog` | 否 |
| 旧 `src/graph` 多入口工作流 | 删除后重写 | 多主链削弱论文叙事 | `scripts/run_case.py` + workflow front-end | 否 |
| 旧 `config/*` 多组兼容配置 | 删除后重写 | 配置层为旧平台服务过多 | 单一 `config/config.yaml` + dataset/experiment groups | 否 |
| 旧 `tests/*` | 删除后重写 | 旧测试编码了 facade 和旧 split 术语 | 新 `tests/unit` + `tests/smoke` | 否 |
| `ref/test`、`labels_ref/tst` 等旧 split 术语 | 删除 | 违反 canonical `train/val/test` 协议 | `train/val/test` | 否 |
| 旧版 `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` 平台叙事 | 删除后重写 | 仍引用旧主链、旧路径和旧运行方式 | `README.md` + 薄适配层 AI 指南 | 否 |
