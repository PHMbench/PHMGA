# PHMGA Graph Workflow System Documentation

This document provides comprehensive guidance for working with the PHMGA LangGraph workflow system.

## Overview

The PHMGA system uses LangGraph to orchestrate multi-agent workflows through two main graph types:

- **Builder Graph**: Constructs the signal processing DAG through iterative planning
- **Executor Graph**: Executes analysis on completed DAGs

## Builder Graph (DAG Construction)

```python
from src.phm_outer_graph import build_builder_graph

builder_graph = build_builder_graph()
# Flow: START → Plan → Execute → Reflect → (Plan | Report) → END
built_state = builder_graph.invoke(initial_state)
```

**Workflow Flow:**
1. **START**: Initialize with PHMState
2. **Plan**: Generate processing plan using plan_agent
3. **Execute**: Apply operators to build DAG using execute_agent
4. **Reflect**: Quality check using reflect_agent
5. **Decision**: Continue planning or proceed to reporting
6. **END**: Complete DAG construction

## Executor Graph (Analysis Execution)

```python
from src.phm_outer_graph import build_executor_graph

executor_graph = build_executor_graph()
# Flow: START → Inquire → Prepare → Train → Report → END
final_state = executor_graph.invoke(built_state)
```

**Workflow Flow:**
1. **START**: Initialize with built DAG
2. **Inquire**: Similarity analysis using inquirer_agent
3. **Prepare**: Dataset preparation using dataset_preparer_agent
4. **Train**: ML training using shallow_ml_agent
5. **Report**: Generate final report using report_agent
6. **END**: Complete analysis results

## Complete Pipeline

```python
def run_complete_analysis(config_path: str):
    """Run complete PHMGA analysis pipeline."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Phase 1: Build DAG
    builder_graph = build_builder_graph()
    initial_state = initialize_state(**config)
    built_state = builder_graph.invoke(initial_state)
    
    # Phase 2: Execute analysis
    executor_graph = build_executor_graph()  
    final_state = executor_graph.invoke(built_state)
    
    return final_state
```

## Graph Configuration

### LangGraph JSON Configuration

```json
{
  "dependencies": ["."],
  "graphs": {
    "builder": "src.phm_outer_graph:build_builder_graph",
    "executor": "src.phm_outer_graph:build_executor_graph"
  },
  "env": ".env"
}
```

### Checkpointing

Graphs support checkpointing for resumable execution:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpointer
checkpointer = SqliteSaver.from_conn_string(":memory:")

# Build graph with checkpointing
builder_graph = build_builder_graph().compile(checkpointer=checkpointer)

# Run with thread ID for resumability
config = {"configurable": {"thread_id": "analysis_session_1"}}
result = builder_graph.invoke(initial_state, config=config)
```

## Error Handling

```python
def robust_graph_execution(graph, initial_state):
    """Execute graph with comprehensive error handling."""
    
    try:
        result = graph.invoke(initial_state)
        return result, None
        
    except Exception as e:
        error_state = initial_state.model_copy(
            update={"error_logs": [f"Graph execution failed: {e}"]}
        )
        return error_state, str(e)
```

## Custom Node Development

```python
def custom_analysis_node(state: PHMState) -> Dict[str, Any]:
    """Custom node for specialized analysis."""
    
    # Perform custom analysis
    results = custom_analysis_logic(state)
    
    # Return state updates
    return {
        "custom_results": results,
        "iteration_count": state.iteration_count + 1
    }

# Add to graph
from langgraph.graph import StateGraph

graph = StateGraph(PHMState)
graph.add_node("custom_analysis", custom_analysis_node)
graph.add_edge("plan", "custom_analysis")
graph.add_edge("custom_analysis", "execute")
```

This graph workflow system provides sophisticated orchestration for multi-agent signal processing workflows in the PHMGA framework.