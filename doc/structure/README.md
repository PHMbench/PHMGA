# PHMGA Structure Docs

## 这套文档服务谁

- 给论文版 PHMGA 的实现者：用统一叙事冻结仓库边界。
- 给复现实验的人：明确数据协议、graph path 和正式产物。
- 给后续删改代码的人：任何目录、类型和脚本都必须能回指到论文主链。

## 当前实现态

这些文档描述的是当前仓库已经落地的最小研究骨架，而不是抽象愿景：

- 根目录 `main.py` 是正式 Hydra 入口。
- `scripts/preflight.py` 与 `scripts/run_case.py` 是由 `main.py` 调用的执行层库模块。
- `config/runs/*.yaml` 是正式 Hydra preset 层。
- `RM_101_THU_GEARBOX` 与 `RM_017_Ottawa19` 已接入统一 canonical protocol。
- `dag_only`、`ml`、`torch` 三条 graph path 都已有最小可运行闭环。
- `torch` path 已切到 graph-level operator PT execution，并使用最小 torch tensor runtime。

## 为什么现在要删库重构

旧分支承载了过重的平台式工程壳，存在多入口、多兼容层、多套 split 语义和并行主链。论文版仓库必须反过来由研究问题驱动，只保留服务主实验闭环的最小研究核心。

## 论文主线是什么

PHMGA 是一个面向工业时间序列的研究框架：前端 agentic workflow 生成 DAG 结构先验，bridge 将 validated DAG JSON 编译到不同 graph path，后端完成训练、评估和报告。

## 如何阅读整套结构文档

1. 先看 `index.md`，确认主链和三条 graph path。
2. 再看 `00` 到 `03`，确认协议、DAG/算子、workflow/bridge、训练评估。
3. 再看 `05_missing_assets_and_roadmap.md`，确认当前缺口、recommended default 和 decision pending。
4. 最后看 `04_rebuild_checklist.md` 和 `del/`，按固定顺序重建并记录删除决策。

## 当前审阅补充

- `del/02_agent_review_findings.md` 是本轮针对 `feature-NSNet` 与 `journal_thesis` 的正式审阅报告。
- 该报告优先回答：当前 agent/prompt/dag/bridge 哪些地方还不符合论文主线，哪些问题必须先修，哪些事项仍是 gap。
