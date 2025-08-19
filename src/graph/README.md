# PHMGA Graph System

The graph module implements the core workflow orchestration for the PHMGA system using LangGraph. It provides two main graph types: the Builder Graph for DAG construction and the Executor Graph for analysis execution.

## Architecture Overview

The PHMGA system uses a dual-graph architecture:

1. **Builder Graph**: Constructs signal processing DAGs through iterative planning, execution, and reflection
2. **Executor Graph**: Executes analysis on completed DAGs through similarity analysis, ML training, and reporting

Both graphs operate on the shared `PHMState` and provide different workflow patterns for different phases of analysis.

## Core Components

### 1. Builder Graph (`build_builder_graph`)

The Builder Graph implements an iterative "think-act-reflect" cycle for constructing signal processing pipelines.

#### Workflow

```
START → Plan → Execute → Reflect → [Plan/Report] → END
```

#### Implementation

```python
def build_builder_graph() -> StateGraph:
    """
    Constructs the graph that builds a computational DAG through iterative planning.
    This graph implements a think-act-reflect cycle for signal processing pipeline construction.
    """
    builder = StateGraph(PHMState)
    
    # Add agent nodes
    builder.add_node("plan", plan_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE"))
    builder.add_node("report", report_agent_node)
    
    # Define workflow edges
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")
    
    # Conditional routing based on reflection
    builder.add_conditional_edges(
        "reflect",
        lambda state: "plan" if state.needs_revision else "report",
        {"plan": "plan", "report": "report"}
    )
    
    builder.add_edge("report", END)
    
    return builder.compile()
```

#### Key Features

- **Iterative Planning**: Continuously refines processing plans based on results
- **Dynamic Execution**: Builds DAG incrementally through operator application
- **Quality Reflection**: Evaluates progress and determines next actions
- **Adaptive Workflow**: Routes between planning and reporting based on analysis quality

### 2. Executor Graph (`build_executor_graph`)

The Executor Graph performs comprehensive analysis on completed DAGs.

#### Workflow

```
START → Inquire → Prepare → Train → Report → END
```

#### Implementation

```python
def build_executor_graph() -> StateGraph:
    """
    Constructs the graph that executes a finalized computational DAG.
    This graph performs similarity analysis, dataset preparation, model training, and reporting.
    """
    builder = StateGraph(PHMState)
    
    # Define analysis pipeline
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine", "euclidean"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", lambda state: {"ml_results": shallow_ml_agent(datasets=state.datasets)})
    builder.add_node("report", report_agent_node)
    
    # Linear execution flow
    builder.add_edge(START, "inquire")
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)
    
    return builder.compile()
```

#### Key Features

- **Similarity Analysis**: Computes comparative metrics between reference and test signals
- **Dataset Preparation**: Assembles ML-ready datasets from processed features
- **Model Training**: Trains and evaluates machine learning models
- **Comprehensive Reporting**: Generates detailed analysis reports

## Usage Examples

### Basic Builder Graph Usage

```python
from src.phm_outer_graph import build_builder_graph
from src.utils import initialize_state

# Create builder graph
builder_graph = build_builder_graph()

# Initialize state
state = initialize_state(
    user_instruction="Analyze bearing signals for fault detection",
    metadata_path="/path/to/metadata.xlsx",
    h5_path="/path/to/signals.h5",
    ref_ids=[1, 2, 3, 4, 5],
    test_ids=[6, 7, 8, 9, 10],
    case_name="bearing_analysis"
)

# Execute builder workflow
final_state = builder_graph.invoke(state)

# Check results
print(f"DAG construction completed with {len(final_state.dag_state.nodes)} nodes")
print(f"Final DAG depth: {get_dag_depth(final_state.dag_state)}")
```

### Basic Executor Graph Usage

```python
from src.phm_outer_graph import build_executor_graph

# Create executor graph
executor_graph = build_executor_graph()

# Use state from builder graph
analysis_state = executor_graph.invoke(final_state)

# Check analysis results
print(f"Similarity analysis completed for {len(analysis_state.dag_state.leaves)} leaf nodes")
print(f"ML models trained: {len(analysis_state.ml_results.get('models', {}))}")
print(f"Final report length: {len(analysis_state.final_report)} characters")
```

### Complete Pipeline

```python
def run_complete_analysis(user_instruction, data_config):
    """Run complete PHMGA analysis pipeline."""
    
    # Phase 1: Build signal processing DAG
    builder_graph = build_builder_graph()
    
    initial_state = initialize_state(
        user_instruction=user_instruction,
        **data_config
    )
    
    built_state = builder_graph.invoke(initial_state)
    
    # Phase 2: Execute analysis on built DAG
    executor_graph = build_executor_graph()
    final_state = executor_graph.invoke(built_state)
    
    return final_state

# Usage
data_config = {
    "metadata_path": "/path/to/metadata.xlsx",
    "h5_path": "/path/to/signals.h5", 
    "ref_ids": [1, 2, 3, 4, 5],
    "test_ids": [6, 7, 8, 9, 10],
    "case_name": "complete_analysis"
}

result = run_complete_analysis(
    "Perform comprehensive bearing fault diagnosis",
    data_config
)
```

## Advanced Features

### Conditional Routing

The Builder Graph uses sophisticated conditional routing:

```python
def should_continue_planning(state: PHMState) -> str:
    """Determine next action based on reflection results."""
    
    if state.needs_revision:
        if state.iteration_count > 5:
            return "report"  # Prevent infinite loops
        else:
            return "plan"    # Continue iterating
    else:
        return "report"      # Analysis complete

# Usage in graph
builder.add_conditional_edges(
    "reflect",
    should_continue_planning,
    {"plan": "plan", "report": "report"}
)
```

### State Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add checkpointing for resumable execution
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

builder_graph = build_builder_graph()
compiled_graph = builder_graph.compile(checkpointer=checkpointer)

# Execute with checkpointing
config = {"configurable": {"thread_id": "analysis_session_1"}}
result = compiled_graph.invoke(state, config=config)
```

### Error Recovery

```python
def error_handling_wrapper(agent_func):
    """Wrapper for agent error handling."""
    
    def wrapped_agent(state):
        try:
            return agent_func(state)
        except Exception as e:
            # Log error and continue
            state.error_logs.append(f"Agent error: {e}")
            return {"error_occurred": True}
    
    return wrapped_agent

# Apply to agents
builder.add_node("plan", error_handling_wrapper(plan_agent))
```

## Graph Visualization

### DAG Visualization

```python
def visualize_graph_execution(state: PHMState):
    """Visualize the constructed DAG."""
    
    tracker = state.tracker()
    
    # Save DAG visualization
    tracker.write_png("dag_visualization.png")
    
    # Export structure
    dag_json = tracker.export_json()
    with open("dag_structure.json", "w") as f:
        f.write(dag_json)
    
    print(f"DAG visualization saved with {len(state.dag_state.nodes)} nodes")
```

### Workflow Monitoring

```python
def monitor_workflow_progress(state: PHMState, stage: str):
    """Monitor workflow execution progress."""
    
    print(f"=== {stage.upper()} STAGE ===")
    print(f"Iteration: {state.iteration_count}")
    print(f"DAG nodes: {len(state.dag_state.nodes)}")
    print(f"DAG depth: {get_dag_depth(state.dag_state)}")
    print(f"Errors: {len(state.error_logs)}")
    print(f"Needs revision: {state.needs_revision}")
    print("=" * 30)

# Add monitoring to graph
def monitored_plan_agent(state):
    monitor_workflow_progress(state, "planning")
    return plan_agent(state)

builder.add_node("plan", monitored_plan_agent)
```

## Integration Patterns

### Custom Agent Integration

```python
def add_custom_analysis_step(builder: StateGraph):
    """Add custom analysis step to workflow."""
    
    def custom_agent(state: PHMState):
        """Custom analysis agent."""
        # Implement custom logic
        return {"custom_results": "analysis_complete"}
    
    # Insert into workflow
    builder.add_node("custom", custom_agent)
    builder.add_edge("execute", "custom")
    builder.add_edge("custom", "reflect")
```

### Multi-Graph Coordination

```python
def coordinate_multiple_analyses():
    """Coordinate multiple analysis workflows."""
    
    builder_graph = build_builder_graph()
    executor_graph = build_executor_graph()
    
    # Run multiple cases
    cases = ["case1", "case2", "case3"]
    results = {}
    
    for case in cases:
        # Build DAG for case
        state = initialize_state_for_case(case)
        built_state = builder_graph.invoke(state)
        
        # Execute analysis
        final_state = executor_graph.invoke(built_state)
        results[case] = final_state
    
    return results
```

## Performance Considerations

### Parallel Execution

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_graph_execution(states):
    """Execute multiple graphs in parallel."""
    
    builder_graph = build_builder_graph()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(builder_graph.invoke, state)
            for state in states
        ]
        
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
    
    return results
```

### Memory Optimization

```python
def memory_efficient_execution(state: PHMState):
    """Execute with memory optimization."""
    
    # Clear intermediate results periodically
    if len(state.dag_state.nodes) > 50:
        # Keep only essential nodes
        essential_nodes = {
            node_id: node for node_id, node in state.dag_state.nodes.items()
            if node_id in state.dag_state.leaves or node.stage == "input"
        }
        state.dag_state.nodes = essential_nodes
    
    return state
```

This graph system provides a robust, flexible foundation for orchestrating complex signal processing and analysis workflows in the PHMGA system.
