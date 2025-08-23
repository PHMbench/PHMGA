# PHMGA Prompt Templates Documentation

This document provides comprehensive guidance for working with the PHMGA prompt system.

## Overview

The prompts module contains carefully crafted prompt templates for different agents in the PHMGA system. These prompts enable effective LLM-agent communication for signal processing tasks.

## Core Prompt Templates

### PLANNER_PROMPT

```python
PLANNER_PROMPT = """You are an expert signal processing engineer creating a detailed processing plan.

**User Instruction:**
{instruction}

**Current DAG State:**
{dag_json}

**Available Tools:**
{tools}

**Previous Reflection (if any):**
{reflection}

**Constraints:**
- Minimum depth: {min_depth}
- Maximum depth: {max_depth}
- Current depth: {current_depth}

**Your Task:**
Create a detailed processing plan as a JSON object with the following structure:
{{
  "plan": [
    {{
      "parent": "node_id",
      "op_name": "operator_name",
      "params": {{}}
    }}
  ]
}}

**Guidelines:**
1. Build upon the current DAG structure
2. Consider the user's instruction and any reflection feedback
3. Ensure logical progression from simple to complex operations
4. Respect depth constraints
5. Use appropriate operators for the signal processing task

**Response Format:** JSON only, no additional text."""
```

### EXECUTE_PROMPT

```python
EXECUTE_PROMPT = """You are an expert signal processing AI, executing a plan.
    
**High-Level Plan:**
{plan}

**Current Data Analysis Graph (DAG):**
{dag}
The current leaf nodes (available for processing) are: {leaves}

**Available Tools:**
{tools}

**Your Task:**
Based on the plan and the current DAG, select the **single next tool** to execute to make progress.
- If the plan involves comparison, ensure you process both reference and test signals before comparing.
- Choose the `parent` for your operation from the available `leaves`.
- Respond with a single JSON object specifying `op_name` and `params`.
- If you believe the plan is fully executed and ready for final insights, respond with {{"op_name": "stop"}}.

**JSON Response:**
"""
```

### REFLECT_PROMPT

```python
REFLECT_PROMPT = """You are a quality assurance expert reviewing a signal processing analysis.

**Original Instruction:**
{instruction}

**Current Processing Stage:**
{stage}

**DAG Overview:**
{dag_overview}

**Issues Encountered:**
{issues_summary}

**Your Task:**
Evaluate the current analysis state and decide on the next action.

**Decision Options:**
- "finish": Analysis is complete and satisfactory
- "need_patch": Minor issues that can be fixed with additional processing
- "need_replan": Major issues requiring a new processing plan
- "halt": Critical errors that prevent continuation

**Response Format:**
{{
  "decision": "finish|need_patch|need_replan|halt",
  "reason": "Detailed explanation of the decision"
}}

**Evaluation Criteria:**
1. Does the DAG achieve the user's instruction?
2. Are there any critical errors or missing components?
3. Is the processing depth and complexity appropriate?
4. Are the results meaningful and interpretable?

Provide your assessment as JSON only."""
```

### REPORT_PROMPT

```python
REPORT_PROMPT = """You are a technical report writer specializing in signal processing and fault diagnosis.

**Analysis Task:**
{instruction}

**DAG Overview:**
{dag_overview}

**Similarity Analysis Results:**
{similarity_stats}

**Machine Learning Results:**
{ml_results}

**Issues Encountered:**
{issues_summary}

**Your Task:**
Generate a comprehensive technical report in Markdown format covering:

1. **Executive Summary**
   - Key findings and recommendations
   - Overall diagnosis confidence

2. **Data Analysis Pipeline**
   - Signal processing steps performed
   - DAG structure and complexity

3. **Similarity Analysis**
   - Comparative metrics between reference and test signals
   - Pattern identification

4. **Machine Learning Results**
   - Model performance metrics
   - Classification accuracy
   - Ensemble results

5. **Conclusions**
   - Final diagnosis
   - Confidence assessment
   - Recommendations

6. **Technical Details**
   - Processing parameters
   - Intermediate results
   - Error analysis (if any)

**Format Requirements:**
- Use proper Markdown formatting
- Include tables for numerical results
- Provide clear section headers
- Be technical but accessible
- Include confidence levels for all conclusions

Generate the complete report now:"""
```

## Usage Patterns

### Using with LangChain

```python
from langchain_core.prompts import ChatPromptTemplate
from src.prompts.plan_prompt import PLANNER_PROMPT

# Create prompt template
prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)

# Format with context
formatted_prompt = prompt.format(
    instruction="Bearing fault diagnosis",
    dag_json=json.dumps(dag_topology),
    tools=tools_description,
    reflection=json.dumps(reflection_history),
    min_depth=4,
    max_depth=8,
    current_depth=2
)
```

### Agent Integration

```python
def plan_agent(state: PHMState) -> dict:
    """Plan agent using structured prompts."""
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    response = (prompt | llm).invoke({
        "instruction": state.user_instruction,
        "dag_json": json.dumps(state.dag_state.dict()),
        "tools": get_tools_description(),
        "reflection": json.dumps(state.reflection_history),
        "min_depth": state.min_depth,
        "max_depth": state.max_depth,
        "current_depth": get_dag_depth(state.dag_state)
    })
    
    return json.loads(response.content)
```

## Prompt Design Principles

### 1. Structured Output

All prompts are designed to produce structured, parseable output:

```python
# Plan prompt produces JSON
{
  "plan": [
    {"parent": "ch1", "op_name": "fft", "params": {}}
  ]
}

# Reflect prompt produces JSON
{
  "decision": "finish",
  "reason": "Analysis complete with satisfactory results"
}
```

### 2. Context Awareness

Prompts include comprehensive context:
- Current system state
- Previous actions and results
- Available tools and constraints
- User instructions and goals

### 3. Role-Based Specialization

Each prompt establishes a clear expert role:
- Signal processing engineer (planning)
- AI execution specialist (execution)
- Quality assurance expert (reflection)
- Technical report writer (reporting)

This prompt system provides the foundation for effective LLM-agent communication throughout the PHMGA analysis pipeline.