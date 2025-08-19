# PHMGA Prompts Module

The prompts module contains carefully crafted prompt templates for different agents in the PHMGA system. These prompts enable effective LLM-agent communication for signal processing tasks.

## Overview

Each agent uses specialized prompts designed for their specific role:
- **Plan Agent**: Goal decomposition and structured planning
- **Execute Agent**: Dynamic operator selection and execution
- **Reflect Agent**: Quality assessment and revision decisions
- **Report Agent**: Comprehensive result analysis and documentation

## Prompt Templates

### 1. Plan Prompt (`plan_prompt.py`)

#### PLANNER_PROMPT

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

#### Usage

```python
from src.prompts.plan_prompt import PLANNER_PROMPT
from langchain_core.prompts import ChatPromptTemplate

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

### 2. Execute Prompt (`execute_prompt.py`)

#### EXECUTE_PROMPT

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

#### Usage

```python
from src.prompts.execute_prompt import EXECUTE_PROMPT

# Used by execute_agent for dynamic operator selection
chain = ChatPromptTemplate.from_template(EXECUTE_PROMPT) | llm
response = chain.invoke({
    "plan": detailed_plan,
    "dag": dag_json,
    "leaves": current_leaves,
    "tools": available_tools
})
```

### 3. Reflect Prompt (`reflect_prompt.py`)

#### REFLECT_PROMPT

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

#### Usage

```python
from src.prompts.reflect_prompt import REFLECT_PROMPT

# Used by reflect_agent for quality assessment
def reflect_agent(instruction, stage, dag_blueprint, issues_summary, state):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REFLECT_PROMPT)
    
    response = (prompt | llm).invoke({
        "instruction": instruction,
        "stage": stage,
        "dag_overview": json.dumps(dag_blueprint),
        "issues_summary": issues_summary or "None"
    })
    
    return json.loads(response.content)
```

### 4. Report Prompt (`report_prompt.py`)

#### REPORT_PROMPT

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

#### Usage

```python
from src.prompts.report_prompt import REPORT_PROMPT

# Used by report_agent for final documentation
def report_agent(instruction, dag_overview, similarity_stats, ml_results, issues_summary):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
    
    response = (prompt | llm).invoke({
        "instruction": instruction,
        "dag_overview": json.dumps(dag_overview),
        "similarity_stats": json.dumps(similarity_stats),
        "ml_results": json.dumps(ml_results),
        "issues_summary": issues_summary or "None"
    })
    
    return {"report_markdown": response.content}
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

### 4. Error Handling

Prompts include guidance for error scenarios:
- Invalid inputs
- Missing information
- Constraint violations
- Recovery strategies

## Advanced Usage Patterns

### Dynamic Prompt Modification

```python
def create_adaptive_prompt(base_prompt, context):
    """Create context-adaptive prompts."""
    
    if context.get("iteration_count", 0) > 3:
        # Add urgency for later iterations
        base_prompt += "\n\nNOTE: This is iteration {iteration_count}. Focus on completing the analysis efficiently."
    
    if context.get("error_count", 0) > 0:
        # Add error awareness
        base_prompt += "\n\nIMPORTANT: Previous errors encountered. Be extra careful with parameter validation."
    
    return base_prompt.format(**context)
```

### Multi-Language Support

```python
PROMPTS = {
    "en": {
        "plan": PLANNER_PROMPT_EN,
        "execute": EXECUTE_PROMPT_EN,
        "reflect": REFLECT_PROMPT_EN,
        "report": REPORT_PROMPT_EN
    },
    "zh": {
        "plan": PLANNER_PROMPT_ZH,
        "execute": EXECUTE_PROMPT_ZH,
        "reflect": REFLECT_PROMPT_ZH,
        "report": REPORT_PROMPT_ZH
    }
}

def get_prompt(agent_type, language="en"):
    """Get localized prompt template."""
    return PROMPTS[language][agent_type]
```

### Prompt Validation

```python
def validate_prompt_output(output, expected_format):
    """Validate LLM output against expected format."""
    
    try:
        if expected_format == "json":
            parsed = json.loads(output)
            return True, parsed
        elif expected_format == "markdown":
            # Basic markdown validation
            return True, output
    except Exception as e:
        return False, f"Validation error: {e}"
```

## Testing and Development

### Prompt Testing

```python
def test_prompt_with_mock_llm():
    """Test prompts with predictable responses."""
    
    from langchain_community.chat_models import FakeListChatModel
    
    mock_responses = [
        '{"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]}',
        '{"decision": "finish", "reason": "Test complete"}',
        '# Test Report\n\nThis is a test report.'
    ]
    
    mock_llm = FakeListChatModel(responses=mock_responses)
    
    # Test each prompt
    for prompt_template in [PLANNER_PROMPT, REFLECT_PROMPT, REPORT_PROMPT]:
        response = mock_llm.invoke(prompt_template.format(**test_context))
        print(f"Response: {response.content}")
```

### Prompt Optimization

```python
def optimize_prompt_performance():
    """Optimize prompts for better LLM performance."""
    
    # Techniques:
    # 1. Clear role definition
    # 2. Structured examples
    # 3. Explicit output format
    # 4. Constraint specification
    # 5. Error handling guidance
    
    optimized_prompt = """
    ROLE: You are an expert signal processing engineer.
    
    TASK: Create a processing plan.
    
    INPUT: {context}
    
    OUTPUT FORMAT: JSON with "plan" array
    
    CONSTRAINTS: 
    - Use only available tools
    - Respect depth limits
    - Follow logical progression
    
    EXAMPLE:
    {"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]}
    
    BEGIN:
    """
    
    return optimized_prompt
```

This prompt system provides the foundation for effective LLM-agent communication throughout the PHMGA analysis pipeline.
