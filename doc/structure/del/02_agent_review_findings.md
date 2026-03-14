# Agent Review Findings

## 审查范围

本报告对照：

- 参考资产：`feature-NSNet`
  - `src/tools/readme.md`
  - `expand / transform / aggregate / multi / decision / signal_processing schemas`
  - `plan / execute / reflect / report / inquirer / research / shared prompts`
- 当前论文版仓库：
  - `src/agents/*`
  - `src/prompts/templates.py`
  - `src/dag/*`
  - `src/bridge/*`
  - `doc/structure/*`

## 结论

当前 `journal_thesis` 已把：

- canonical data protocol
- validated DAG JSON
- graph-path-specific bridge outputs
- 最小 artifact 落盘

这些后端合同搭起来了。

但前端 agent/prompt 主链仍未达标，尤其是：

- `plan_agent` 没有输出结构化计划
- `execute_agent` 没有严格消费计划
- prompts 仍是占位文本

因此，本轮代码重构优先级必须是：

`P0: prompt + StepPlan + execute contract`

而不是继续扩更多训练壳。

## Findings

| ID | 问题 | 根因 | 风险等级 | 修复优先级 | 修复建议 |
| --- | --- | --- | --- | --- | --- |
| `F01` | `execute_agent` 直接按固定模板生成 DAG，没有消费结构化计划 | `WorkflowState.plan` 只是 `List[str]`，不存在结构化 step plan | `Critical` | `P0` | 定义 `StepPlan`，再把 `execute_agent` 改成 plan executor |
| `F02` | `plan_agent` 只返回字符串列表，无法约束 parent / op_name / params | prompt/agent 合同尚未正式化 | `Critical` | `P0` | 定义 `StepPlan` schema，并让 `plan_agent` 成为唯一计划入口 |
| `F03` | `src/prompts/templates.py` 只是四句占位文本，缺少输入字段、输出字段、禁止事项 | 文档先于实现这一步还没做完 | `High` | `P0` | 拆分为 `plan / execute / reflect / report` 正式 prompt 文件 |
| `F04` | `reflect_agent` 只按节点数返回字符串判断，无法输出结构性修复意见 | 未定义 `ReflectionResult` | `High` | `P1` | 定义 `ReflectionResult`，输出 `missing_operators`、`shape_risks`、`whether_replan_needed` |
| `F05` | `DagNode` 缺少 `plan_step_ref`、`legal_paths`、`input_bindings` 等 richer contract 字段 | 当前 DAG IR 只够最小 bridge，不够 plan-driven generation | `High` | `P1` | 先在文档冻结字段，再增量扩充 schema |
| `F06` | bridge 默认假设单父线性链，无法自然支持 `MULTI_VARIABLE` / `DECISION` | `_lineage()` 逻辑按单父倒推 | `High` | `P1` | 未来 bridge 直接读取节点合同，不再从单链反推语义 |
| `F07` | `OperatorCatalog` 只覆盖极小子集，无法承载论文版 prompt planning | 当前算子系统只为 smoke 与合同验证服务 | `Medium` | `P1` | 先迁移 `filter / hilbert_envelope / psd / kurtosis / crest_factor / band_power / concatenate` |
| `F08` | `report_agent` 还没有充分区分 `dag_only / ml / torch` 的报告结构 | 当前报告模板以统一拼接为主 | `Medium` | `P2` | 在 artifacts contract 稳定后再细化 graph-dependent reporting |
| `F09` | `OfflineLLM` 仍是 stub，不能代表真实 provider-backed generation | prompts 与 schema 尚未冻结 | `Medium` | `P2` | 等 `StepPlan / prompts / DagJson` 冻结后再切 OpenRouter |

## 参考分支中最值得迁移的研究资产

### 算子资产

真正有价值的是：

- 五类算子边界
- shape 语义
- 参数 schema 思想
- tool legality 约束

不应整包迁移的是：

- 旧工程壳
- 与当前论文主线无关的 decision/comparison 编排

### prompt 资产

最值得迁移的思想是：

- `plan_prompt` 的 shape-aware planning
- `reflect_prompt` 的结构审查维度
- `shared/render.py` 的轻量模板渲染方式

最不应直接迁移的是：

- 让 `execute_agent` 自己决定“下一步 tool”的自治逻辑
- 与 web research 相关的 prompt 主链

## 风险排序

### `P0`

- `StepPlan` 缺失
- placeholder prompts
- `execute_agent` 绕过计划

### `P1`

- richer DAG contract 缺失
- bridge 对 multi-input 不友好
- operator catalog 过窄

### `P2`

- graph-dependent report 还不够细
- provider-backed LLM 尚未接入
- `torch` path 仍是 NumPy fallback

## Recommended Default

1. 先冻结文档里的 `StepPlan / ReflectionResult / prompt contracts`
2. 再重写 `prompts/`
3. 再改 `execute_agent`
4. 再扩第一批算子
5. 最后才切 provider-backed LLM 与真实 torch-side 训练器

## 本轮已确认的修复方向

- 运行时 planner 输出改为 NVTA 风格 `StepPlan`
- `graph_path` 回收到 config/runtime，不作为 planner 显式输入
- `execute_agent` 负责把代表性计算结果保存回 state
- 四个主路径 agent 都需要单独的输入输出单测
