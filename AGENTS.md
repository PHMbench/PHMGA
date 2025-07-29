

---

# AGENTS.md: PHM-GA 智能体开发规范

本文档定义了 PHM-GA (Prognostics and Health Management Graph Agent) 项目中四大核心智能体的功能、接口和实现逻辑。本文档旨在作为 AI 编程助手（Codex）的开发指导，以确保生成的代码符合项目的双层图（外层静态工作流 + 内层动态数据流图）架构。

## 核心架构概览

系统由一个静态的**外层图 (Outer Graph)** 控制，该图按顺序调度四个核心智能体，形成一个完整的“思考-行动-反思”循环。

**工作流**: `START` -> `Plan` -> `Execute` -> `Reflect` -> (`Plan` 或 `Report`) -> `END`

## 1. Planner Agent (规划智能体)

**文件路径**: plan_agent.py

### 职责

**目标分解器 (Goal Decomposer)**：将用户输入的、模糊的自然语言指令，分解为一系列清晰、高级、可执行的子目标。

### 接口规范

-   **输入**: `state: PHMState`
    -   主要使用 `state.user_instruction` (用户原始指令)
    -   以及 `state.reflection_history` (来自 Reflector Agent 的历史反馈)
-   **输出**: `PHMState`
    -   更新 `state.high_level_plan` 字段。

### 内部逻辑 (Codex 指导)

1.  **构建 Prompt**: 创建一个发送给 LLM 的 Prompt，该 Prompt 必须包含以下部分：
    *   **角色扮演**: "You are an expert PHM (Prognostics and Health Management) engineer. Your task is to break down a high-level user request into a clear, step-by-step analysis plan."
    *   **用户指令**: 插入 `state.user_instruction`。
    *   **历史反馈 (可选)**: 如果 `state.reflection_history` 不为空，则插入历史反馈，并指示 LLM：“Based on the previous feedback, please refine or create a new plan.”
    *   **输出格式要求**: "Provide the plan as a numbered list of high-level goals. Each goal should be a concise, actionable instruction. Do not include any preamble or explanation."

2.  **调用 LLM**: 将构建好的 Prompt 发送给 LLM。

3.  **解析输出**:
    *   接收 LLM 返回的文本。
    *   将文本按行分割，移除数字编号和多余的空格，得到一个字符串列表。
    *   将这个列表赋值给 `state.high_level_plan`。

4.  **返回状态**: 返回更新后的 `state` 对象。

### 伪代码实现

```python
def plan_agent(state: PHMState) -> PHMState:
    # 1. 构建 Prompt
    prompt = f"""
    As a PHM expert, break down the following user request into a step-by-step plan.
    User Request: "{state.user_instruction}"
    """
    if state.reflection_history:
        prompt += f"\nConsider this feedback from previous runs: {state.reflection_history[-1]}"
    
    # 2. 调用 LLM
    llm_response = call_llm(prompt) # 假设的 LLM 调用函数
    
    # 3. 解析输出并更新状态
    plan_steps = parse_llm_response_to_list(llm_response)
    state.high_level_plan = plan_steps
    
    # 重置反思标志，为下一次执行做准备
    state.needs_revision = False 
    
    return state
```

## 2. Executor Agent (执行智能体)

**文件路径**: execute_agent.py

### 职责

**动态执行器 (Dynamic Executor)**：实现一个“思考-行动”循环，在循环的每一步都调用 LLM 来决策下一个要执行的具体信号处理算子，并使用 `DAGTracker` 记录操作。

**多智能体并行执行**：

系统支持根据 `high_level_plan` 同步或异步地调度多个 Executor Agent，每个智能体可独立执行不同的子目标，分别生成各自的 DAG 及分析结果。最终，这些结果可在后续阶段进行整合与汇总，以提升处理效率和分析的全面性。

### 接口规范

-   **输入**: `state: PHMState`
-   **输出**: `PHMState`
    -   `state.dag_state` 将被极大地丰富，包含新的 节点对应于如下工具模块中的操作：
      - `src/tools/aggregate_schemas.py`
      - `src/tools/decision_schemas.py`
      - `src/tools/expand_schemas.py`
      - `src/tools/transform_schemas.py`
      - `src/tools/multi_schemas.py`

### 内部逻辑 (Codex 指导)

1.  **初始化**:
    *   获取 `DAGTracker` 实例: `tracker = state.tracker()`。
    *   获取可用工具列表: 从 `OP_REGISTRY` 中提取所有算子的 `op_name` 和 `description`。

2.  **实现内部循环**: 使用 `for` 循环，设定一个最大执行步数（如 `max_steps=10`）以防止无限循环。

3.  **在循环的每一步**:
    *   **Observe**: 调用 `tracker.export_json()` 获取当前 DAG 的紧凑 JSON 表示。
    *   **Think (调用 LLM)**:
        *   **构建 Prompt**: 创建一个专门的 Prompt，包含：
            *   **角色扮演**: "You are a signal processing expert. Your goal is to select the next best action to achieve the high-level plan."
            *   **高级计划**: 插入 `state.high_level_plan`。
            *   **当前数据状态 (DAG)**: 插入 `tracker.export_json()` 的输出。
            *   **可用工具**: 插入格式化后的工具列表。
            *   **任务指令**: "Based on the plan and the current data, select the **single next tool** to use. Respond with a JSON object containing `op_name` and `params`. If you think the plan is complete, respond with `{\"op_name\": \"stop\"}`."
        *   **调用 LLM**: 发送 Prompt 并获取返回的 JSON 字符串。
    *   **Act**:
        *   解析返回的 JSON。如果 `op_name` 是 `"stop"`，则 `break` 循环。
        *   根据 `op_name` 从 `OP_REGISTRY` 查找并实例化算子。
        *   **确定输入节点**: 从 `state.dag_state.leaves` 中找到当前需要处理的叶子节点 ID。
        *   **加载数据**: 使用节点 ID 从 `state.dag_state.nodes` 中获取节点对象，并加载其数据。
        *   **执行算子**: 调用算子的 `execute()` 方法。
        *   **记录到 DAG**: 调用 `tracker.add_execution()`，将操作的输入、算子本身和输出结果记录到 DAG 中。

4.  **返回状态**: 返回执行循环后更新的 `state`。

## 3. Reflector Agent (反思智能体)

**文件路径**: reflect_agent.py

### 职责

**全局审视者 (Global Reviewer)**：在执行阶段完成后，审视整个数据处理流图（DAG），判断当前获得的信息是否足以做出最终决策。

### 接口规范

-   **输入**: `state: PHMState`
-   **输出**: `PHMState`
    -   更新 `state.needs_revision` (布尔值)
    -   更新 `state.reflection_history` (字符串列表)

### 内部逻辑 (Codex 指导)

1.  **获取 DAG 摘要**: 调用 `state.tracker().export_json()`。

2.  **检查终止条件**:
  *   聚合分析多个 DAG 的有效性和当前分析结果，取长补短，进行高级研究并优化，判断是否需要新建一个 DAG（即重新规划和执行），还是可以进入报告生成阶段（执行 Reporter Agent）。
  *   具体判断标准：如果分析结果不足以支持最终诊断，则设置 `state.needs_revision = True`，并准备新一轮规划；如果分析结果已充分，则设置 `state.needs_revision = False`，进入报告生成流程。


3.  **调用 LLM 进行反思**:
    *   **构建 Prompt**:
        *   **角色扮演**: "You are a quality assurance expert for a data analysis pipeline."
        *   **用户目标**: 插入 `state.user_instruction`。
        *   **当前成果 (DAG)**: 插入 DAG 的 JSON 摘要。
        *   **任务指令**: "Based on the user's goal and the analysis performed so far, is the information sufficient to make a final diagnosis? Answer with a JSON object containing two keys: `is_sufficient` (boolean) and `reason` (string, explaining your thought process and what is missing if not sufficient)."
    *   **调用 LLM** 并解析返回的 JSON。

4.  **更新状态**:
    *   如果 `is_sufficient` 为 `True`，设置 `state.needs_revision = False`。
    *   如果 `is_sufficient` 为 `False`，设置 `state.needs_revision = True`，并将 `reason` 字符串追加到 `state.reflection_history` 中。

5.  **返回状态**: 返回更新后的 `state`。

## 4. Reporter Agent (报告智能体)

**文件路径**: report_agent.py

### 职责

**最终报告生成器 (Final Report Generator)**：当反思智能体确认信息充足后，该智能体负责整合所有分析结果，生成一份图文并茂的、人类可读的最终诊断报告。

### 接口规范

-   **输入**: `state: PHMState`
-   **输出**: `PHMState`
    -   更新 `state.final_report` 字段。

### 内部逻辑 (Codex 指导)

1.  **生成 DAG 可视化**: 调用 `state.tracker().write_png("final_dag.png")`，将最终的 DAG 保存为图片。

2.  **获取 DAG 摘要**: 调用 `state.tracker().export_json()`。

3.  **调用 LLM 生成报告**:
    *   **构建 Prompt**:
        *   **角色扮演**: "You are a senior PHM engineer writing a final diagnostic report for a client."
        *   **原始任务**: 插入 `state.user_instruction`。
        *   **分析证据 (DAG)**: 插入 DAG 的 JSON 摘要。
        *   **任务指令**: "Based on the evidence in the analysis graph, write a comprehensive, clear, and professional diagnostic report in Markdown format. Start with a clear conclusion, then provide a step-by-step justification referencing the operations in the graph. Finally, include the visualization of the workflow using this Markdown syntax: `!Analysis Workflow`."
    *   **调用 LLM** 获取 Markdown 格式的报告文本。

4.  **更新状态**: 将 LLM 生成的报告字符串赋值给 `state.final_report`。

5.  **返回状态**: 返回更新后的 `state`。