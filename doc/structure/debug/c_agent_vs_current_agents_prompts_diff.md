# C_Agent 与当前仓库的 Agents / Prompts 差异分析

## 参考源

本对比只基于以下两组目录：

- 参考仓库：
  - `/home/user/LQ/C_Agent/PHMGA/src/agents`
  - `/home/user/LQ/C_Agent/PHMGA/src/prompts`
- 当前仓库：
  - `src/agents`
  - `src/prompts`

本文件不是逐行源码 diff，而是面向当前论文版实现的差异总览。重点回答：

- 当前仓库相对 C_Agent 删除了什么
- 当前仓库保留了什么
- 当前仓库重写了什么
- 哪些差异是论文版有意收敛
- 哪些差异仍然是未闭合缺口

---

## 1. 结论摘要

当前仓库相对 C_Agent 的核心变化是：

**从 `PHMState + tool schema + file-saving execution` 的状态驱动平台执行体系，收敛为 `WorkflowState + StepPlan + validated DAG JSON + bridge` 的论文版合同驱动前端。**

这意味着当前仓库做了四个方向的收缩：

- `plan_agent` 不再输出旧 `detailed_plan`，而是输出运行时主合同 `StepPlan`
- `execute_agent` 不再是自治 executor，也不直接在 agent 内部生成正式磁盘 artifact
- `reflect_agent` 从只返回 `decision/reason` 的轻结构，升级为 `ReflectionResult`
- `report_agent` 从“最终诊断叙述”收敛为“graph-dependent artifacts 驱动的报告器”

同时，当前仓库明确没有继承 C_Agent 的额外平台主链：

- `dataset_preparer_agent`
- `inquirer_agent`
- `shallow_ml_agent`

这些不属于当前论文版主路径。

---

## 2. 目录级差异

### C_Agent 独有

#### agents

- `dataset_preparer_agent.py`
- `inquirer_agent.py`
- `shallow_ml_agent.py`

#### prompts

- `reflector_prompt.py` 以外的主路径 prompt 文件虽然当前仓库也有，但语义已显著改写

### 当前仓库独有

#### prompts

- `shared.py`
  - 负责统一渲染 prompt contract header
- `templates.py`
  - 仅作为兼容层存在，不再是正式 prompt 入口

### 差异判断

- `dataset_preparer / inquirer / shallow_ml` 在 C_Agent 中属于平台式外围 agent
- 当前仓库有意删去这类外围主链，只保留 `plan / execute / reflect / report`
- 当前仓库新增 `shared.py` 是为了把 prompt 明确合同化，而不是继续扩工程壳

---

## 3. 四个主路径 agent 差异

### `plan_agent`

#### C_Agent

保留要点：

- 基于 LangChain `ChatPromptTemplate`
- 直接绑定旧 `PHMState`
- 通过 `tools description` 和 `dag_topology` 生成 `detailed_plan`
- 输出字段是 `{"detailed_plan": [...]}`，不是正式运行时类型

#### 当前仓库

重写要点：

- 基于 `WorkflowState`
- 先构建 `SignalContext`
- 输出 `StepPlan`
- `graph_path` 由 config/runtime 隐式提供，不作为 planner 的显式输入
- planner 只更新 workflow state，不直接触达训练或模型参数

#### 差异结论

- `保留`：NVTA 风格 `parent/op_name/params` 规划思想
- `重写`：状态模型、输出合同、graph_path 注入方式
- `删除`：旧 `PHMState + detailed_plan` 绑定
- `尚缺`：多轮 replan 状态机和 richer planner repair 机制

### `execute_agent`

#### C_Agent

保留要点：

- 从 `detailed_plan` 逐步执行
- 通过 `get_operator()` 和 `tool schema` 查算子
- 直接处理 `InputData / ProcessedData`
- 支持 `MultiVariableOp`
- 在 agent 内部直接落盘 `.npy/.npz`

#### 当前仓库

重写要点：

- 只消费 `StepPlan`
- 结果写回 `WorkflowState.execution_results`
- 输出 `validated DAG JSON`
- 把不能执行的步骤显式写成 `ExecutionGap`
- decision 节点当前只作为 auxiliary terminal，不进入正式执行链
- 磁盘 artifact 落盘交给 `scripts/run_case.py` 和后端层

#### 差异结论

- `保留`：逐步执行 plan、缺参补全、multi-input 最小支持
- `重写`：执行语义、状态写回位置、artifact 边界
- `删除`：自治式下一步决定、agent 内部直接产出正式文件结果
- `尚缺`：dataset-level execution、richer multi-parent legality、更强 shape validation

### `reflect_agent`

#### C_Agent

保留要点：

- 输入是 `instruction / stage / dag_blueprint / issues_summary`
- 核心输出是 `decision/reason`
- 依赖旧 `PHMState` 与 `dag_state`

#### 当前仓库

重写要点：

- 输出正式类型 `ReflectionResult`
- 除 `decision/reason` 外，还显式输出：
  - `missing_operators`
  - `shape_risks`
  - `structural_warnings`
- 结果写回 `reflection_results` 与 `reflection_history`

#### 差异结论

- `保留`：NVTA 风格 `decision/reason` 审查模式
- `重写`：输出结构化程度、状态承载方式
- `删除`：对旧 `PHMState` / `dag_state` 的直接依赖
- `尚缺`：更丰富的修复建议与多轮 replan 闭环

### `report_agent`

#### C_Agent

保留要点：

- 输入偏向：
  - `dag_overview`
  - `similarity_stats`
  - `ml_results`
- 输出是报告 markdown
- 更偏“诊断总结 + 相似度 + 模型评估”叙述

#### 当前仓库

重写要点：

- 输入偏向：
  - `compiled_manifest`
  - `path_artifacts`
  - `reflection_summary`
  - `review_context`
- 明确按 `dag_only / ml / torch` 生成 graph-dependent sections
- `report_agent` 本身不生产 artifact，只消费 artifact

#### 差异结论

- `保留`：最终用 LLM 或模板汇总证据形成 markdown 的思想
- `重写`：证据来源、graph-dependent 报告结构、artifact 边界
- `删除`：旧的 similarity/ML-only 报告中心
- `尚缺`：更强的 evidence richness 和论文附录级表达

---

## 4. 主路径 prompt 差异

### `plan_prompt`

#### C_Agent

- 更强调 PHM 域操作多样性
- 更强调从任意已有节点扩 DAG
- 更强调 envelope / time-frequency / cross-channel 等丰富策略
- 更接近“领域策略型 planner prompt”

#### 当前仓库

- 保留 NVTA 风格的 `StepPlan` 输出格式
- 新增明确的合同头：
  - 输入字段
  - 输出字段
  - 禁止事项
- prompt 目标收敛为“产生可执行的结构化下一步计划”

#### 差异判断

- `保留`：PHM 域规划思路
- `重写`：prompt contract 化
- `尚缺`：更丰富的 operators roadmap 进入当前 planner prompt

### `execute_prompt`

#### C_Agent

- 语义是：根据 high-level plan、当前 DAG 和 leaves，选择“下一步单个工具”
- 本质上仍给 executor 留有自治决策空间

#### 当前仓库

- 语义是：只把 `step_plan` materialize 成 DAG node updates
- 明确禁止：
  - 发明新的 plan steps
  - 静默跳过 unsupported step
  - 修改 training/model internals

#### 差异判断

- 这是当前差异最大的 prompt 之一
- `删除`：executor 自治选择下一步工具
- `重写`：从“决策 prompt”改成“执行合同 prompt”

### `reflect_prompt`

#### C_Agent

- 重点围绕 `decision/reason`
- 更偏架构评论和高层审查

#### 当前仓库

- 保留 `decision/reason`
- 明确冻结结构化输出字段
- 明确禁止用训练指标替代结构审查

#### 差异判断

- `保留`：审查角色和决策导向
- `重写`：输出合同显式化

### `report_prompt`

#### C_Agent

- 更偏最终诊断、相似度洞察和 ML 结果叙述
- 报告中心是诊断结论

#### 当前仓库

- 更偏 graph-dependent artifact reporting
- graph path 不同，报告段落也不同
- 明确禁止虚报 torch 能力或引用不存在 artifact

#### 差异判断

- `保留`：基于输入证据生成 markdown
- `重写`：报告中心从“诊断叙述”转向“artifact evidence reporting”

---

## 5. 有意收敛 vs 仍然缺失

### 有意收敛

以下差异应被视为论文版有意收敛，而不是能力回退：

- `删除` 旧 `PHMState`
- `删除` executor 自治决策
- `删除` agent 内部直接写磁盘结果
- `删除` 旧平台额外 agent 主链
- `重写` prompts 为合同化 prompts，而不是纯模板文本
- `重写` 报告输入为 `compiled_manifest + path_artifacts`

### 仍然缺失

以下差异说明当前仓库还没有把论文版主链补完：

- 尚缺多轮 orchestration
- 尚缺 dataset-level execution
- 尚缺 richer operator coverage
- 尚缺更强的 report evidence richness
- 尚缺 `decision` 节点进入正式执行链或正式外环 artifact 的稳定设计

---

## 6. 对当前仓库的直接结论

### 应保持的差异

- 当前仓库应继续保持 `WorkflowState + StepPlan + validated DAG JSON + bridge` 主链
- 当前仓库应继续保持 `execute_agent` 是 plan-driven materializer，不是自治 executor
- 当前仓库应继续保持 `report_agent` 只消费 artifacts，不生产 artifacts
- 当前仓库应继续保持 prompts 的合同化写法

### 应记为 TODO 的差异

- dataset-level execution
- 多轮 `plan -> execute -> reflect -> replan`
- richer operator catalog
- richer report evidence chain
- `decision` 节点的正式地位

### 不应迁回的 C_Agent 资产

- `dataset_preparer_agent`
- `inquirer_agent`
- `shallow_ml_agent`
- 旧 `PHMState`
- executor 自治式“选下一步工具”逻辑
- agent 内部直接保存 `.npy/.npz` 作为正式结果的做法

---

## 最终判断

当前仓库相对 C_Agent 的方向不是“功能更少”，而是：

**去掉平台式执行壳，把前端主链收敛成论文版合同驱动前端。**

因此，这份对比的核心结论不是“把 C_Agent 原样搬回来”，而是：

- 哪些资产已经被正确吸收
- 哪些平台逻辑被有意删除
- 哪些论文主链缺口还需要继续补齐
