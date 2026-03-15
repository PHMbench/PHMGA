# 05 Missing Assets And Roadmap

本文件记录当前论文版主链的现状、尚缺和推荐下一步。写法按主链阶段分组，而不是按文件散列。

## signal / protocol

### current_status

- canonical protocol 已固定为 `train/val/test`
- real + synthetic 两类数据入口已统一到 `src/data/protocol.py`
- `SignalContext` 已从 representative preview signal 构建

### missing_now

- dataset-level DAG execution 仍未实现
- preview signal 与 full split execution 的边界仍需继续强化

### recommended_next

- 继续保持前端只消费 `SignalContext`
- 后续单独补 dataset-level materializer，而不是把 raw windows 直接塞进 planner

## plan

### current_status

- `plan_agent` 已输出 `StepPlan`
- prompt 已经读取 richer operator summary
- `graph_path` 不再作为 planner 显式输入

### missing_now

- richer operator coverage 仍不足
- provider-backed planner 还未接成默认主链

### recommended_next

- 先扩 operator metadata 和少量 PHM 高频算子
- 等 prompt/contract 稳定后再接真实 provider

## execute

### current_status

- `execute_agent` 已是 plan-driven materializer
- 已支持单输入链和最小 `multi.concatenate`
- 参数补全顺序已固定为：
  - `StepPlan.params`
  - context-derived values
  - operator defaults
  - LLM tuning for `llm_tunable_params`

### missing_now

- `decision` 仍未进入正式可执行链
- multi-parent lineage 的后端支持仍然较弱

### recommended_next

- 保持 `decision` 先作为 auxiliary terminal
- 等 bridge 升级后再扩 richer multi-parent execution

## reflect

### current_status

- `reflect_agent` 已输出结构化 `ReflectionResult`
- `need_patch / need_replan / finish / halt` 语义已经进入状态机

### missing_now

- reflection 仍主要看结构合同，不看 richer evidence quality

### recommended_next

- 后续把 artifact richness 和 operator diversity 纳入 reflection 规则

## bridge / path artifacts

### current_status

- `validated DAG JSON` 仍是唯一法定接口
- `compiled_dag_manifest.json` 已稳定
- `ml / torch` 路径已接入 `dataset_preparer`、`shallow_ml`、`inquirer`

### missing_now

- bridge 对 multi-parent lineage 仍不是一等支持
- `decision` 仍没有正式 compiled side-output plan

### recommended_next

- 先保持 bridge 主链最小可解释
- 再补 richer lineage 和 decision-side compilation

## report

### current_status

- `report_agent` 已按 `dag_only / ml / torch` 消费 graph-dependent artifacts
- similarity artifacts 已进入 `ml / torch` 报告证据链

### missing_now

- 报告仍偏实验记录，不是论文附录级 evidence report
- provider-backed summarization 仍未成为默认主链

### recommended_next

- 先扩 evidence richness
- 再考虑 provider-backed 摘要生成

## 当前正式目标

以下项已从“纯缺口”转为当前正式目标：

- richer operator schema metadata
- `dataset_preparer` 进入 `src/data`
- `inquirer / shallow_ml` 进入 `src/model`
- multi-round `replan` 状态机
- execute 阶段对 operator params 的 LLM tuning

## Decision Pending

只记录真正影响主链的未决项：

- `decision` 节点何时进入正式可执行链
- bridge 何时升级到 richer multi-parent lineage
- provider-backed planner / reflector / reporter 何时接成默认
