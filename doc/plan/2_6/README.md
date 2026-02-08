# 2.6 现状总结：LLM 生成 DAG → DAG2TSPN → 端到端训练（已跑通）与缺口清单

本目录用于沉淀 **2.5 计划落地后的“当前可运行现状”**，并明确指出：
> “LLM 生成 DAG 并把 DAG 转换为 TSPN，完成训练/报告”已经跑通到了哪一步；  
> 若要形成真正的 **闭环结构搜索（外环）↔ 可微训练（内环）**，还欠缺哪些关键环节。

关联文档（规范与接口是 source of truth）：
- Contract（Tensor/FFT/对齐等硬约束）：`doc/plan/2_5/SPEC.md`
- IO 协议（TrainReport/ConfigPatch）：`doc/plan/2_5/AGENT_IO.md`
- 路线图（Phase 0/1/...）：`doc/plan/2_5/PLAN.md`

本目录文件：
- 现状（已实现/已通过测试/如何运行）：`doc/plan/2_6/STATUS.md`
- 欠缺环节（LLM→DAG→TSPN 闭环还缺什么）：`doc/plan/2_6/GAPS.md`
- 快速运行与测试开关（conda/pytest/env）：`doc/plan/2_6/RUNBOOK.md`

