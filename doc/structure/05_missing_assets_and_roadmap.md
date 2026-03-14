# 05 Missing Assets And Roadmap

本文件不是功能说明，而是当前论文版仓库尚未补齐的关键合同清单。

## 当前缺失资产

### prompts

当前正式 prompt 文件已经补到：

- `plan_prompt.py`
- `execute_prompt.py`
- `reflect_prompt.py`
- `report_prompt.py`
- `shared.py`

仍未补齐的是：

- provider-backed 真正调用链
- prompt output repair 策略
- 更丰富的 decision / inquiry prompt 分支

### agent contracts

当前 `WorkflowState` 已补到：

- `signal_context`
- `step_plan`
- `execution_results`
- `execution_gaps`
- `reflection_history: List[str]`
- `reflection_results`
- `dag`

当前仍待补齐：

- 多轮 replan 状态机
- richer `decision` 执行语义
- path-specific report section contract 进一步细化

### DAG contract

当前 `DagNode` 已补到：

- `operator_category`
- `legal_paths`
- `input_bindings`
- `plan_step_ref`
- `rationale`

仍待补齐：

- `shape_symbols` 等更细的符号级 shape 推演
- multi-parent lineage 在 bridge 中的一等支持

### operator system

当前 `OperatorCatalog` 只覆盖：

- `signal.normalize`
- `signal.fft_mag`
- `feature.mean`
- `feature.std`
- `feature.rms`

这只能证明 bridge 路径打通，不能证明论文方法在 PHM 信号处理上有足够的结构表达力。

## 推荐默认

### prompts

- 主路径只保留 `plan / execute / reflect / report`
- `inquirer` 只作为 decision path 扩展，不进入第一轮主实验闭环
- `research/shared` 不进入主路径，只保留为外部研究辅助资产

### agent contracts

- `plan_agent` 输出 `StepPlan`
- `execute_agent` 只能消费 `StepPlan`
- `reflect_agent` 输出 `ReflectionResult`
- `report_agent` 只消费 graph-dependent artifacts 与 manifest

### DAG contract

- 继续坚持 `validated DAG JSON` 是前后端唯一法定接口
- 继续坚持 `NetworkX` 只用于生成期，不进入 bridge 输出合同
- 继续坚持 schema 校验失败时拒绝进入 bridge，不做隐式修复

### operator roadmap

第一批应优先补齐：

- `signal.filter`
- `signal.hilbert_envelope`
- `signal.psd`
- `signal.stft`
- `signal.wavelet_transform`
- `feature.kurtosis`
- `feature.crest_factor`
- `feature.band_power`
- `feature.spectral_centroid`
- `multi.concatenate`

## Decision Pending

以下事项当前应显式记录，而不是在代码里默默脑补：

- `multi-variable` 节点是否只允许无名 `parents`，还是要正式引入 `input_bindings`
- `decision` 节点是否进入正式 DAG JSON，还是只作为 `dag_only` / 报告外环 artifact
- `torch` path 中哪些传统算子需要可微代理，哪些只保留固定前端语义
- `report_agent` 是否需要 provider-backed 摘要生成，还是继续以模板化报告为主

## LLM backend roadmap

当前后端状态是：

- `OfflineLLM`
- `mode = offline_stub`

这是合同验证态，不是最终论文态。切换到 provider-backed OpenRouter 的前提必须全部满足：

1. `plan / execute / reflect / report` prompt contract 冻结
2. `StepPlan / ReflectionResult / DagJson` schema 冻结
3. `compiled_dag_manifest.json` 与 graph-dependent artifact contract 冻结
4. 至少一个真实数据集上的真实生成案例进入测试
5. 对 LLM 输出解析失败、schema repair、replan 触发条件给出显式策略

## Path maturity matrix

| path | 当前成熟度 | 当前可用性 | 主要缺口 | 推荐下一步 |
| --- | --- | --- | --- | --- |
| `dag_only` | `M2: contract-usable` | 可导出合法 DAG 与 manifest，且已切到 `StepPlan` 驱动 | 仍缺真实 provider-backed 生成链 | 先稳住 prompts / tests / execute contract |
| `ml` | `M1: baseline-usable` | 最小 ML 闭环可跑 | 只支持线性单父链特征；缺少 richer operators | 先扩一批统计与频域算子 |
| `torch` | `M0: placeholder` | artifact contract 可验证 | 仍是 NumPy fallback，不是正式 torch 训练栈 | 等 prompts/DAG/operator 合同稳定后再升级 |

## 文档优先顺序

后续代码改动前，必须先审过以下文档：

1. `del/02_agent_review_findings.md`
2. `01_dag_and_operators.md`
3. `02_workflow_and_bridge.md`
4. `03_training_and_evaluation.md`
