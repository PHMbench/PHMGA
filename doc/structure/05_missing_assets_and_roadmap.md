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
- 五类 operator schema 已进入 planner 可见范围
- `graph_path` 不再作为 planner 显式输入

### missing_now

- richer operator coverage 仍不足
- OpenRouter free-model provider 仍未形成可替代 Formal Main 的稳定 tuple
- `offline_stub` 继续只保留为 pilot / deterministic baseline

### recommended_next

- 先把 fixed compiled/runtime graph 跑稳
- 再稳住 Codex formal-main 路径与 OpenRouter candidate qualification 的错误处理、回归测试和文档
- 然后再扩 operator metadata 和少量 PHM 高频算子

## execute

### current_status

- `execute_agent` 已是 plan-driven materializer
- 已支持 richer single-input chain、`multi.concatenate`、`multi.cross_correlation`
- `decision.threshold` 已进入 terminal side-output 半执行态
- 参数补全顺序已固定为：
  - `StepPlan.params`
  - context-derived values
  - operator defaults
  - LLM tuning for `llm_tunable_params`

### missing_now

- multi-parent lineage 的后端支持仍然较弱
- richer decision family 仍未扩展，当前只有最小阈值式 terminal node

### recommended_next

- 保持 `decision` 先作为 terminal side-output
- 等 bridge 升级后再扩 richer multi-parent execution 与 richer decision family

## reflect

### current_status

- `reflect_agent` 已输出结构化 `ReflectionResult`
- `need_patch / need_replan / finish / halt` 语义已经进入状态机
- `dag_quality_evaluator` 已能输出当前 round 的紧凑质量摘要
- provider mode 下已可走 OpenRouter-backed reflector

### missing_now

- `dag_quality_evaluator` 仍是最小版本，还没有 richer operator-diversity 与增量收益判断
- reflection 仍未消费 dataset-level execution 证据

### recommended_next

- 继续保持 `dag_quality_evaluator` 为最小合同
- 后续再把 artifact richness 和 operator diversity 纳入 reflection 规则

## bridge / path artifacts

### current_status

- `validated DAG JSON` 仍是唯一法定接口
- `compiled_dag_manifest.json` 已稳定
- `ml / torch` 路径已接入 `dataset_preparer`、`shallow_ml`、`inquirer`
- `torch` path 已切到 graph-level operator PT execution，并使用最小 tensor runtime
- `ml / torch` compiled plan 已切到：
  - `execution_nodes`
  - `output_specs`
  - `output_policy`
- `dag_quality_summary.json` 已进入正式 artifact 列表
- `decision_side_outputs.json` 已进入正式 artifact 列表

### missing_now

- `decision` 目前仍没有 richer compiled side-output plan，只有最小 terminal evidence payload
- `torch` trainer 仍是最小线性头，不是 richer trainable stack
- `output_policy` 目前只支持：
  - `terminal_only`
  - `include_intermediate_features`
- `GraphModule / module factory / learnable control` 已有最小实现，但还没有形成研究级 trainer 与 planner 默认主链
- `WaveFilters / Ricker / Chirplet / Laplace / Morlet` 已进入统一 operator/runtime 路线，但仍缺默认 planner 采用与系统化 ablation

### recommended_next

- 先稳住 multi-parent compiled lineage 与 output policy contract
- 固定阶段顺序为：
  - `phase_1_fixed_compiled_runtime`
  - `phase_2_provider_backed_llm`
  - `phase_3_module_runtime`
  - `phase_4_learnable_control`
- 再补 richer decision-side compilation、ablation ledger 和 torch trainer
- 后续 torch runtime 再逐步过渡到：
  - `compiled execution plan`
  - `GraphModule / module factory`
  - optional gate / attention controls
- `WaveFilters` family 已统一归入 `src/operators/transform_ops.py`
- 未来参数策略继续统一走：
  - `OperatorSpec.param_schema`
  - `OperatorSpec.param_defaults`
  - `OperatorSpec.param_docs`
  - `OperatorSpec.llm_tunable_params`
- 不新增单独 `wavefilters.*` config 面

## report

### current_status

- `report_agent` 已按 `dag_only / ml / torch` 消费 graph-dependent artifacts
- similarity artifacts 已进入 `ml / torch` 报告证据链
- `dag_quality_summary` 已进入报告的简短质量段落
- `report_agent` 已统一为 `ChatPromptTemplate | llm` 风格，provider transport 仍集中在 `src/llm/client.py`
- root 默认配置下仍保留 deterministic renderer；formal main 已冻结为 Codex-backed provider reporter

### missing_now

- 报告仍偏实验记录，不是论文附录级 evidence report
- provider-backed report 的真实网络稳定性仍需继续积累 smoke 与 ledger 证据

### recommended_next

- 先扩 evidence richness，并继续保持 root default 的 deterministic baseline
- formal main 继续用 Codex-backed reporter；后续重点转向 OpenRouter qualification、smoke、回归测试和 incident note，而不是再回退到“未接通 provider”

## 当前正式目标

以下项已从“纯缺口”转为当前正式目标：

- Hydra root config + `main.py` 统一入口
- richer operator schema metadata
- `dataset_preparer` 进入 `src/data`
- `inquirer / shallow_ml` 进入 `src/model`
- multi-round `replan` 状态机
- execute 阶段对 operator params 的 LLM tuning
- compact `dag_quality_evaluator`
- 五类 operator schema 的 richer metadata 与首轮 runnable subset
- multi-parent compiled support 的正式设计边界：
  - `plan_agent` 继续只生成方法 DAG
  - bridge 负责 path-specific compiled output 选择
  - `decision` 继续只做 side-output
- 固定 graph first 的阶段顺序：
  - 先 fixed compiled/runtime
  - 再 provider-backed LLM
  - 再 GraphModule / module factory
  - 最后 gate / attention / learnable control

## Decision Pending

只记录真正影响主链的未决项：

- `decision` 节点何时进入正式可执行链
- bridge 何时升级到 richer multi-parent lineage
- provider-backed planner / reflector / reporter 何时接成默认
- OpenRouter candidate 何时能稳定到足以替换当前 Codex formal-main tuple
- torch runtime 何时从当前最小实现升级到更强 trainer / batch runtime / richer module stack
- `GraphModule + learnable control` 何时进入论文主表而不是增强实验
