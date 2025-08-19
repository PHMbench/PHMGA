# PHMGA Agents Module

The agents module contains the core intelligent agents that form the backbone of the PHMGA (Prognostics and Health Management Graph Agent) system. These agents implement a "think-act-reflect" cycle for automated signal processing and fault diagnosis.

## Architecture Overview

The PHMGA system uses a dual-layer architecture:
- **Outer Graph**: Static workflow orchestrating agent execution
- **Inner Graph**: Dynamic signal processing DAG constructed by agents

**Workflow**: `START` → `Plan` → `Execute` → `Reflect` → (`Plan` or `Report`) → `END`

## Core Agents

### 1. Plan Agent (`plan_agent.py`)

**Purpose**: Goal decomposition and high-level planning

The Plan Agent transforms user instructions into structured, executable plans using LLM-powered analysis.

#### API Reference

```python
def plan_agent(state: PHMState) -> dict:
    """
    Generate a detailed processing plan using structured output.

    Parameters
    ----------
    state : PHMState
        Current system state containing:
        - user_instruction: Natural language task description
        - reflection_history: Previous feedback from reflect agent
        - dag_state: Current DAG topology
        - min_depth, max_depth: Planning constraints

    Returns
    -------
    dict
        Dictionary with 'detailed_plan' key containing list of Step objects:
        - parent: str - Parent node ID to apply operation
        - op_name: str - Signal processing operator name
        - params: dict - Operator parameters

    Raises
    ------
    Exception
        Planning errors are captured and logged to state.error_logs
    """
```

#### Data Structures

```python
class Step(BaseModel):
    """A single step in the processing plan."""
    parent: str = Field(..., description="Parent node ID")
    op_name: str = Field(..., description="Operator name from OP_REGISTRY")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operator parameters")

class Plan(BaseModel):
    """The complete processing plan."""
    plan: List[Step] = Field(..., description="Sequence of processing steps")
```

#### Usage Example

```python
from src.agents.plan_agent import plan_agent
from src.states.phm_states import PHMState

# Initialize state with user instruction
state = PHMState(
    user_instruction="Analyze bearing signals for fault detection",
    dag_state=dag_state,
    reference_signal=ref_signal,
    test_signal=test_signal
)

# Generate plan
result = plan_agent(state)
detailed_plan = result["detailed_plan"]

# Plan contains structured steps
for step in detailed_plan:
    print(f"Apply {step['op_name']} to {step['parent']} with {step['params']}")
```

#### Integration

- **Input Dependencies**: User instruction, DAG topology, operator registry
- **Output Usage**: Execute agent consumes the detailed plan
- **LLM Integration**: Uses structured output for reliable plan generation
- **Error Handling**: Graceful degradation with error logging

### 2. Execute Agent (`execute_agent.py`)

**Purpose**: Dynamic signal processing pipeline construction

The Execute Agent implements a think-act loop, iteratively selecting and applying signal processing operators to build the computational DAG.

#### API Reference

```python
def execute_agent(state: PHMState) -> Dict[str, Any]:
    """
    Execute signal processing operations based on the detailed plan.

    Parameters
    ----------
    state : PHMState
        Current system state containing:
        - detailed_plan: List of operations to execute
        - dag_state: Current DAG with nodes and leaves
        - case_name: For organizing saved outputs

    Returns
    -------
    Dict[str, Any]
        Updated state with:
        - dag_state: Enhanced DAG with new nodes
        - executed_steps: Number of operations performed

    Notes
    -----
    - Supports both single-variable and multi-variable operators
    - Automatically resolves missing parameters using LLM
    - Saves intermediate results to disk
    - Updates DAG topology immutably
    """
```

#### Execution Logic

1. **Plan Processing**: Iterates through detailed_plan steps
2. **Operator Resolution**: Retrieves operators from OP_REGISTRY
3. **Parameter Resolution**: Uses LLM to fill missing required parameters
4. **Data Execution**: Applies operators to reference and test signals
5. **DAG Updates**: Creates new nodes and updates topology
6. **Result Persistence**: Saves outputs to case-specific directories

#### Multi-Variable Support

```python
# Single-variable operators (most common)
if not issubclass(op_cls, MultiVariableOp):
    out_ref, out_tst = _execute_single_variable_op(op, parent_id, nodes)

# Multi-variable operators (comparison, arithmetic)
else:
    out_ref, out_tst = _execute_multi_variable_op(op, parent_ids, nodes)
```

#### Usage Example

```python
from src.agents.execute_agent import execute_agent

# State with plan from plan_agent
state.detailed_plan = [
    {"parent": "ch1", "op_name": "fft", "params": {}},
    {"parent": "fft_ch1", "op_name": "mean", "params": {"axis": -2}}
]

# Execute the plan
result = execute_agent(state)

# Check results
print(f"Executed {result['executed_steps']} operations")
print(f"New DAG has {len(result['dag_state'].nodes)} nodes")
```

#### Error Handling

- **Invalid Operators**: Logs errors and continues execution
- **Missing Parameters**: Automatically resolved via LLM
- **Data Mismatches**: Graceful handling with error logging
- **File I/O Issues**: Continues execution without persistence

### 3. Reflect Agent (`reflect_agent.py`)

**Purpose**: Quality assessment and revision decisions

The Reflect Agent evaluates the constructed DAG and determines whether the analysis is sufficient or requires revision.

#### API Reference

```python
def reflect_agent(
    *,
    instruction: Optional[str] = None,
    stage: Optional[str] = None,
    dag_blueprint: Optional[Dict[str, Any]] = None,
    issues_summary: Optional[str] = None,
    state: PHMState
) -> Dict[str, str]:
    """
    Quality check the DAG and return a decision with reason.

    Parameters
    ----------
    instruction : str, optional
        Original user instruction
    stage : str, optional
        Current processing stage (e.g., "POST_PLAN", "POST_EXECUTE")
    dag_blueprint : dict, optional
        JSON representation of current DAG
    issues_summary : str, optional
        Summary of errors encountered
    state : PHMState
        Complete system state for context

    Returns
    -------
    Dict[str, str]
        Decision dictionary:
        - decision: str - One of {"finish", "need_patch", "need_replan", "halt"}
        - reason: str - Explanation for the decision
    """

def reflect_agent_node(state: PHMState, *, stage: str) -> Dict[str, Any]:
    """
    Adapter for the outer graph using PHMState.

    Parameters
    ----------
    state : PHMState
        Current system state
    stage : str
        Processing stage identifier

    Returns
    -------
    Dict[str, Any]
        State updates:
        - needs_revision: bool - Whether revision is needed
        - reflection_history: List[str] - Updated history
    """
```

#### Decision Logic

The reflect agent evaluates multiple factors:

- **DAG Completeness**: Sufficient depth and breadth
- **Error Analysis**: Issues encountered during execution
- **Goal Achievement**: Alignment with user instruction
- **Quality Metrics**: Signal processing pipeline effectiveness

#### Valid Decisions

- `"finish"`: Analysis complete, proceed to reporting
- `"need_patch"`: Minor fixes needed, continue execution
- `"need_replan"`: Major issues, restart planning
- `"halt"`: Critical errors, stop execution

#### Usage Example

```python
from src.agents.reflect_agent import reflect_agent_node

# After execution phase
result = reflect_agent_node(state, stage="POST_EXECUTE")

if result["needs_revision"]:
    print("Revision needed:", result["reflection_history"][-1])
    # Return to planning or execution
else:
    print("Analysis complete, proceeding to report")
    # Continue to reporting
```

### 4. Inquirer Agent (`inquirer_agent.py`)

**Purpose**: Similarity analysis between reference and test signals

The Inquirer Agent calculates similarity metrics between processed signals to enable comparative analysis.

#### API Reference

```python
def inquirer_agent(state: PHMState, metrics: List[str]) -> Dict[str, List[str]]:
    """
    Calculate similarity matrix between reference and test signals.

    Parameters
    ----------
    state : PHMState
        System state with processed signals in leaf nodes
    metrics : List[str]
        Similarity metrics to compute:
        - "cosine": Cosine similarity
        - "euclidean": Euclidean distance
        - "pearson": Pearson correlation distance

    Returns
    -------
    Dict[str, List[str]]
        Result summary:
        - "new_nodes": List of node IDs with updated similarity data

    Notes
    -----
    - Updates node.sim attribute with similarity matrices
    - Processes all leaf nodes in the DAG
    - Handles multi-dimensional feature vectors
    """
```

#### Similarity Metrics

```python
def _calc_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    """Calculate similarity between two feature vectors."""
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    elif metric == "euclidean":
        return float(np.linalg.norm(a - b))
    elif metric == "pearson":
        r = np.corrcoef(a, b)[0, 1]
        return float(1 - r)
```

#### Usage Example

```python
from src.agents.inquirer_agent import inquirer_agent

# After signal processing is complete
result = inquirer_agent(state, metrics=["cosine", "euclidean"])

# Check similarity results
for leaf_id in state.dag_state.leaves:
    node = state.dag_state.nodes[leaf_id]
    if hasattr(node, 'sim') and node.sim:
        print(f"Node {leaf_id} similarities: {node.sim}")
```

### 5. Dataset Preparer Agent (`dataset_preparer_agent.py`)

**Purpose**: Feature extraction and dataset assembly for machine learning

The Dataset Preparer Agent collects processed features and assembles them into training and test datasets with proper labels.

#### API Reference

```python
def dataset_preparer_agent(
    state: PHMState,
    *,
    config: Dict | None = None
) -> Dict:
    """
    Gather features and assemble datasets with labels.

    Parameters
    ----------
    state : PHMState
        System state with processed nodes containing features
    config : dict, optional
        Configuration options:
        - stage: str - Node stage to process (default: "processed")

    Returns
    -------
    Dict
        Dataset collection:
        - datasets: Dict[str, Dict] - Node ID to dataset mapping
        - n_nodes: int - Number of datasets created

    Dataset Structure
    ----------------
    Each dataset contains:
    - X_train: np.ndarray - Training features
    - X_test: np.ndarray - Test features
    - y_train: np.ndarray - Training labels
    - y_test: np.ndarray - Test labels
    - origin_node: str - Source node ID
    """
```

#### Label Resolution

```python
def _find_root_labels(node_id: str, all_nodes: Dict) -> Dict[str, Any]:
    """
    Traverse DAG upward to find root node labels.

    Returns the labels dictionary from the root node that contains
    the ground truth classifications for all samples.
    """
```

#### Usage Example

```python
from src.agents.dataset_preparer_agent import dataset_preparer_agent

# After feature extraction
result = dataset_preparer_agent(state)

datasets = result["datasets"]
for node_id, dataset in datasets.items():
    print(f"Node {node_id}:")
    print(f"  Training: {dataset['X_train'].shape} features, {len(dataset['y_train'])} labels")
    print(f"  Testing: {dataset['X_test'].shape} features, {len(dataset['y_test'])} labels")
```

### 6. Shallow ML Agent (`shallow_ml_agent.py`)

**Purpose**: Machine learning model training and ensemble inference

The Shallow ML Agent trains traditional machine learning models on extracted features and performs ensemble classification.

#### API Reference

```python
def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train models per dataset and perform ensemble inference.

    Parameters
    ----------
    datasets : Dict[str, Dict[str, Any]]
        Dataset collection from dataset_preparer_agent
    algorithm : str, default="RandomForest"
        ML algorithm: "RandomForest" or "SVM"
    ensemble_method : str, default="hard_voting"
        Ensemble strategy: "hard_voting" or "soft_voting"
    cv_folds : int, default=5
        Cross-validation folds for model evaluation

    Returns
    -------
    Dict[str, Any]
        Training results:
        - models: Dict[str, Dict] - Trained models with metrics
        - ensemble_metrics: Dict - Ensemble performance
        - metrics_markdown: str - Formatted results table

    Model Structure
    --------------
    Each model entry contains:
    - metrics: Dict - accuracy, f1, cv_accuracy, cv_f1, etc.
    - model_b64: str - Base64 encoded model for persistence
    """
```

#### Supported Algorithms

```python
def _build_estimator(algorithm: str) -> Any:
    """Return a scikit-learn estimator based on algorithm."""
    if algorithm.upper() == "SVM":
        return SVC(probability=True, random_state=42)
    return RandomForestClassifier(random_state=42)
```

#### Ensemble Methods

- **Hard Voting**: Majority vote from high-quality models (CV accuracy > 90%)
- **Soft Voting**: Average probability predictions from models with `predict_proba`

#### Usage Example

```python
from src.agents.shallow_ml_agent import shallow_ml_agent

# Train models on prepared datasets
results = shallow_ml_agent(
    datasets=state.datasets,
    algorithm="RandomForest",
    ensemble_method="soft_voting",
    cv_folds=5
)

# Examine results
print("Individual Model Performance:")
for node_id, model_data in results["models"].items():
    metrics = model_data["metrics"]
    print(f"  {node_id}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")

print(f"\nEnsemble Performance:")
print(f"  Accuracy: {results['ensemble_metrics']['accuracy']:.3f}")
print(f"  F1 Score: {results['ensemble_metrics']['f1']:.3f}")

# Display formatted table
print("\nDetailed Metrics:")
print(results["metrics_markdown"])
```

### 7. Report Agent (`report_agent.py`)

**Purpose**: Comprehensive result analysis and documentation

The Report Agent generates detailed markdown reports summarizing the entire analysis pipeline and results.

#### API Reference

```python
def report_agent(
    *,
    instruction: str,
    dag_overview: Dict[str, Any],
    similarity_stats: Dict[str, Any],
    ml_results: Dict[str, Any],
    issues_summary: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate a final markdown report via LLM.

    Parameters
    ----------
    instruction : str
        Original user instruction
    dag_overview : Dict[str, Any]
        JSON representation of the complete DAG
    similarity_stats : Dict[str, Any]
        Similarity analysis results from inquirer_agent
    ml_results : Dict[str, Any]
        Machine learning results from shallow_ml_agent
    issues_summary : str, optional
        Summary of errors encountered during processing

    Returns
    -------
    Dict[str, str]
        Report content:
        - report_markdown: str - Complete markdown report
    """

def report_agent_node(state: PHMState) -> Dict[str, str]:
    """
    Adapter for the outer graph using PHMState.

    Parameters
    ----------
    state : PHMState
        Complete system state with all analysis results

    Returns
    -------
    Dict[str, str]
        Final report:
        - final_report: str - Comprehensive markdown report

    Side Effects
    -----------
    - Saves final DAG visualization to disk
    - Extracts similarity stats from leaf nodes
    """
```

#### Report Sections

The generated report typically includes:

1. **Executive Summary**: High-level findings and recommendations
2. **Data Overview**: Input signals and preprocessing steps
3. **Signal Processing Pipeline**: DAG visualization and operator sequence
4. **Similarity Analysis**: Comparative metrics between reference and test signals
5. **Machine Learning Results**: Model performance and ensemble predictions
6. **Conclusions**: Final diagnosis and confidence assessment
7. **Technical Details**: Processing parameters and intermediate results

#### Usage Example

```python
from src.agents.report_agent import report_agent_node

# Generate comprehensive report
result = report_agent_node(state)
final_report = result["final_report"]

# Save report to file
with open("analysis_report.md", "w") as f:
    f.write(final_report)

print(f"Generated report with {len(final_report)} characters")
```

## Agent Integration Patterns

### State Management

All agents operate on the shared `PHMState` object:

```python
from src.states.phm_states import PHMState

# State contains all necessary context
state = PHMState(
    user_instruction="Bearing fault diagnosis",
    dag_state=dag_state,
    reference_signal=ref_signal,
    test_signal=test_signal,
    detailed_plan=[],
    reflection_history=[],
    # ... other fields
)
```

### Error Handling

Agents use consistent error handling patterns:

```python
try:
    # Agent operation
    result = perform_operation()
except Exception as e:
    # Log error and continue gracefully
    state.dag_state.error_log.append(f"Agent error: {e}")
    result = default_result()
```

### LLM Integration

Agents that use LLMs follow standard patterns:

```python
from src.model import get_llm
from src.configuration import Configuration

# Get configured LLM instance
llm = get_llm(Configuration.from_runnable_config(None))

# Use structured output for reliability
structured_llm = llm.with_structured_output(OutputSchema)
result = structured_llm.invoke(prompt_data)
```

## Testing

Each agent includes comprehensive test cases:

```python
if __name__ == "__main__":
    # Test with mock data
    test_state = create_test_state()
    result = agent_function(test_state)
    assert_expected_behavior(result)
    print("✅ Agent test passed!")
```

### Running Tests

```bash
# Test individual agents
python -m src.agents.plan_agent
python -m src.agents.execute_agent
python -m src.agents.reflect_agent

# Run full test suite
python -m pytest tests/test_*_agent.py
```

## Dependencies

### Required Packages

- **Core**: `numpy`, `scipy`, `pydantic`
- **LLM**: `langchain-core`, `langchain-google-genai`
- **ML**: `scikit-learn`, `pandas`, `joblib`
- **Visualization**: `matplotlib`, `networkx`

### Internal Dependencies

- `src.states.phm_states`: State management
- `src.tools.signal_processing_schemas`: Operator registry
- `src.model`: LLM factory functions
- `src.configuration`: System configuration
- `src.prompts.*`: Agent-specific prompts

## Architecture Integration

### Outer Graph Workflow

```python
from src.phm_outer_graph import build_builder_graph, build_executor_graph

# Builder graph: Plan → Execute → Reflect (loop)
builder_graph = build_builder_graph()

# Executor graph: Inquire → Prepare → Train → Report
executor_graph = build_executor_graph()
```

### Data Flow

1. **Input**: Raw signals and user instruction
2. **Planning**: LLM generates structured processing plan
3. **Execution**: Operators applied to build signal processing DAG
4. **Reflection**: Quality assessment and revision decisions
5. **Analysis**: Similarity computation and dataset preparation
6. **Learning**: Model training and ensemble inference
7. **Reporting**: Comprehensive result documentation

### Extensibility

The agent system is designed for easy extension:

```python
# Add new agent
def custom_agent(state: PHMState, **kwargs) -> Dict[str, Any]:
    """Custom analysis agent."""
    # Implement custom logic
    return {"custom_results": results}

# Register in outer graph
builder.add_node("custom", custom_agent)
builder.add_edge("reflect", "custom")
```

This modular design enables rapid prototyping and deployment of new analysis capabilities while maintaining system coherence and reliability.
```
