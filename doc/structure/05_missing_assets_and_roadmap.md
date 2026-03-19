# 05 Missing Assets And Roadmap

本文件只记录当前论文版 PHMGA 的主线缺口与推荐顺序。  
总原则固定为：

**先锁死 Agent Core，再补 Dataset-Level Evidence，最后做 Comparison Layer。**

这意味着：

- `validated DAG JSON -> compile_dag_for_path()` 继续是唯一法定接口
- `DECISION` 继续保持 terminal side-output，不进入 `ml / torch` 训练张量主链
- root `config/config.yaml` 继续只是 smoke/development baseline，不承担 paper mainline 语义
- canonical diagnosis backend 当前固定为 `ml`
- `torch`、provider candidate、runtime enhancement 继续只处在比较层

## M0: Agent Core

### current_status

- `PHMState -> StateGraph(plan -> execute -> dag_quality -> reflect -> rollback|compile_ready)` 已成立
- `plan_agent` 已输出结构化 `StepPlan`
- `execute_agent` 已是 plan-driven materializer，并显式记录 `ExecutionGap`
- `reflect_agent` 已输出 `finish / need_patch / need_replan / halt`
- `validated DAG JSON` 仍是 bridge 前唯一法定接口
- 当前 richer operator schema 已覆盖五类语义：
  - `EXPAND`
  - `TRANSFORM`
  - `AGGREGATE`
  - `MULTI_VARIABLE`
  - `DECISION`

### missing_now

- Agent core 的正式验收口径还需要持续收紧到：
  - `plan_agent` 产出稳定 `StepPlan`
  - `execute_agent` 不新增计划外步骤
  - `rollback` 真正恢复 `last_stable_dag` 与 `last_stable_execution_results`
  - `reflect_agent` 在有限轮内稳定收敛
  - 每轮都能导出 compileable 的 `validated DAG JSON`
- richer operator coverage 仍不足，但这里只补**阻塞 agent 主线**的高频 PHM 算子

### recommended_next

- 把 M0 继续固定为最短因果链：
  - `protocol + signal_context`
  - `PHMState / StateGraph`
  - `plan -> execute -> dag_quality -> reflect -> finish|rollback`
  - `validated DAG JSON`
  - `bridge`
  - `ml` path 最小 compiled execution sanity
- 暂停让 provider/path/runtime 复杂度干扰 M0 验收
- 不新增第二套 workflow、bridge 或目录重构；`configuration.py` 和 `phm_outer_graph.py` 继续只作为 compat/transition layer 叙述

## M1: Dataset-Level Evidence

### current_status

- `dag_quality_summary.json` 已开始承载 split-level sampled dataset evidence
- sampled evidence pass 已从 `train/val/test` sampled windows 中补充：
  - DAG 是否可 materialize
  - feature / multi 输出是否非空、有限、维度一致
  - 最小 class separation / proxy probe evidence
  - `decision` terminal side-output 的 split-level summary
- `reflect_agent` 和 `report_agent` 已开始把 sampled dataset evidence 当成比 preview 更强的事实源

### missing_now

- full dataset execution 仍未实现；当前仍是 split-level sampled evidence pass
- reflection 对 dataset-level evidence 的利用还可以更稳定、更可解释
- report 仍需继续增强 dataset-level diagnosis evidence richness

### recommended_next

- 保持 planner 只消费 `SignalContext`，不把 raw windows 直接塞进前端
- 继续把 dataset-level 证据固定挂在：
  - `dag_quality_summary.dataset_level`
- 只回答最小必要问题：
  - train / val / test 上是否可 materialize
  - feature / multi / decision side-output 是否稳定、非空、可区分
  - proxy probe 最小监督证据是否成立
- 不新建第二套 workflow；dataset-level evidence 只是主线增强

## M2: Comparison Layer

### current_status

- `dag_only / ml / torch` 三条 path 已并存
- formal main 已冻结到 Codex canonical transport
- OpenRouter candidate 已降到 qualification 语境
- `GraphModule / module factory / learnable control` 已有最小实现
- `WaveFilters / Ricker / Chirplet / Laplace / Morlet` 已进入统一 operator/runtime 路线

### missing_now

- comparison layer 的文档和实验还需要持续防止“反向定义主线”
- `torch` trainer 仍是最小线性头，不是 research-grade stack
- provider candidate 仍未形成可替代 canonical mainline 的稳定 tuple

### recommended_next

- comparison 层只比较：
  - `dag_only / ml / torch`
  - Codex canonical transport 与 OpenRouter candidate qualification
  - `output_policy`
  - `module_runtime`
  - `learnable_control`
- 不让 backend/provider/path 成为“PHMGA 核心是什么”的解释中心
- `torch` 继续诚实定位为比较层最小实现
- `WaveFilters` family 继续按统一 operator/runtime 路线推进，不新增单独 config 面
- 未来参数策略继续统一走：
  - `OperatorSpec.param_schema`
  - `OperatorSpec.param_defaults`
  - `OperatorSpec.param_docs`
  - `OperatorSpec.llm_tunable_params`

## 暂停优先做的方向

在 M0 和 M1 闭合前，不继续优先做：

- `DECISION` 主链化
- 更重的 `torch` trainer / batch runtime / richer module stack
- task-level backend routing
- richer provider 竞争
- 大批新 runtime family 默认化
- 以“后端炫技”为导向扩 operator family

## Decision Pending

只记录真正影响主线边界的未决项：

- sampled dataset evidence pass 是否已经足以支撑当前论文主张
- `decision` 节点何时从 side-output 升级为正式可执行链
- bridge 何时升级到更强的 richer multi-parent lineage
- OpenRouter candidate 何时能稳定到足以进入更强 comparison，而不是只做 qualification
- `torch` runtime 何时从当前最小实现升级到更强 trainer / batch runtime / richer module stack
- `GraphModule + learnable control` 何时进入论文主表而不是增强实验
