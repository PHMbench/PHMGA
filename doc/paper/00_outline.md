# Paper Outline

## 1. Introduction

- 问题：agentic PHM workflow 缺少可执行 bridge 与可审计实验链
- 贡献：
  - validated DAG JSON
  - compiled execution plan
  - `ml / torch` graph paths
  - artifact-based experiment loop

## 2. System

- `protocol -> PHMState / StateGraph -> plan / execute / dag_quality / reflect / rollback|compile_ready -> validated DAG JSON -> bridge -> compiled plan -> path runtime`
- `ChatPromptTemplate | llm` 的前端 agent 编排与 `compile_dag_for_path()` 的法定边界
- `offline_stub` 与 provider-backed LLM 的角色边界

## 3. Method

- richer PHM DAG
- multi-parent compiled support
- output policy
- GraphModule / learnable runtime 作为增强层

## 4. Experiments

- 主结果：
  - Ottawa
  - RM101
  - `ml` vs `torch`
- 消融：
  - output policy
  - module runtime / learnable control
  - WaveFilters
  - provider mode

### Figures / Tables

- Figure 1: system pipeline
- Figure 2: validated DAG / compiled subgraph example
- Table 1: Ottawa / RM101 main results
- Table 2: output policy / runtime ablation
- Table 3: provider ablation
- Appendix: rejected experiments / ledger summary

## 5. Analysis

- DAG evidence
- StateGraph / rollback evidence
- report evidence
- failure cases
- keep / reject ledger

## 6. Limitations

- `decision` 仍未进入训练主链
- planner 尚未主动使用 GraphModule / WaveFilters / learnable control
- WaveFilters 与 learnable runtime 仍需更系统的正式实验
