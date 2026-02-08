# 2.5 可解释网络（Unified-X / TSPN）在 PHMGA 的集成计划

本目录用于存放“把你在 `Unified_X_fault_diagnosis/` 中的可解释网络（Signal_processing / Feature_extract / TSPN）迁移并融入 PHMGA，并用外环智能体自动生成/修改 `model_config.yaml` 控制网络结构”的工程计划与不可变约定。

- 不可变约定（Contract）：`doc/plan/2_5/SPEC.md`
- 外环/内环 agent 输入输出与 Patch 规范：`doc/plan/2_5/AGENT_IO.md`
- 分阶段落地路线图（验收标准/测试用例）：`doc/plan/2_5/PLAN.md`

> 说明：PHMGA 当前已存在一个 `src/model/explainable/` 的 TSPN 子实现，并且 `load_tspn_config()` 已提供对 `config_basic.yaml` 的“legacy adapter”。  
> 本计划的重点是：补齐 Unified 版本的算子/特征覆盖、参数协议、解释性证据结构化输出，以及把“结构搜索（外环）/训练评估（内环）”闭环跑稳、跑可复现。

