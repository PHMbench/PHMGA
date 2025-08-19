# PHMGA State Management System

The states module provides the core data structures and state management for the PHMGA system. It implements a sophisticated state management architecture that tracks both the computational graph (DAG) and the overall system state throughout the analysis pipeline.

## Architecture Overview

The state management system consists of three main layers:

1. **Node Layer**: Individual data nodes representing signals and processing results
2. **DAG Layer**: Graph topology and relationships between nodes
3. **System Layer**: Overall system state including configuration and metadata

## Core Components

### 1. Node Types

#### Base Node Class (`_NodeBase`)

All nodes inherit from the base node class which provides common functionality:

```python
class _NodeBase(BaseModel):
    """Base class for all DAG nodes."""

    node_id: str = Field(default_factory=lambda: f"n_{uuid.uuid4().hex[:8]}")
    parents: List[str] | str  # Parent node IDs
    stage: Literal["input", "processed", "similarity", "dataset", "output"] = "input"
    shape: Shape  # Data dimensions
    kind: Literal["signal"] = "signal"
    sim: Dict[str, Any] = Field(default_factory=dict, description="Similarity metrics")
```

**Key Features:**
- **Unique IDs**: Automatically generated unique identifiers
- **Parent Tracking**: Maintains DAG topology through parent references
- **Stage Classification**: Tracks processing stage for workflow management
- **Shape Information**: Preserves dimensional metadata
- **Similarity Storage**: Stores comparative analysis results

#### Input Data Nodes (`InputData`)

Represent raw input signals at the root of the DAG:

```python
class InputData(_NodeBase):
    """Raw input signal node."""

    stage: Literal["input"] = "input"
    data: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, np.ndarray] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
```

**Purpose:**
- Store original signal data
- Maintain metadata (sampling frequency, labels, etc.)
- Serve as DAG root nodes

**Usage Example:**
```python
from src.states.phm_states import InputData
import numpy as np

# Create input node for channel 1
ch1_node = InputData(
    node_id="ch1",
    parents=[],
    shape=(1, 1024, 2),  # (batch, length, channels)
    data={"signal": raw_signal_data},
    results={
        "ref": reference_signals,  # Reference dataset
        "tst": test_signals       # Test dataset
    },
    meta={
        "fs": 1000,  # Sampling frequency
        "channel": "accelerometer_x",
        "labels": {"id1": "healthy", "id2": "faulty"}
    }
)
```

#### Processed Data Nodes (`ProcessedData`)

Represent the results of signal processing operations:

```python
class ProcessedData(_NodeBase):
    """Processed signal node."""

    stage: Literal["processed"] = "processed"
    source_signal_id: str
    method: str
    processed_data: np.ndarray | None = None
    results: Dict[str, np.ndarray] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
```

**Purpose:**
- Store processed signal results
- Track processing method and parameters
- Maintain links to source signals

**Usage Example:**
```python
# Create FFT processing node
fft_node = ProcessedData(
    node_id="fft_ch1",
    parents=["ch1"],
    shape=(1, 513, 2),  # FFT output shape
    source_signal_id="ch1",
    method="fft",
    results={
        "ref": fft_reference_results,
        "tst": fft_test_results
    },
    meta={
        "tool": "fft",
        "params": {"axis": -2},
        "parent": "ch1"
    }
)
```

#### Similarity Nodes (`SimilarityNode`)

Store similarity analysis results between reference and test signals:

```python
class SimilarityNode(_NodeBase):
    """Similarity analysis node."""

    stage: Literal["similarity"] = "similarity"
    source_node_id: str
    metrics: List[str]
    similarity_matrix: Dict[str, Dict[str, Dict[str, float]]] = Field(default_factory=dict)
```

**Purpose:**
- Store comparative analysis results
- Support multiple similarity metrics
- Enable classification and diagnosis

**Usage Example:**
```python
# Create similarity analysis node
sim_node = SimilarityNode(
    node_id="sim_fft_ch1",
    parents=["fft_ch1"],
    source_node_id="fft_ch1",
    metrics=["cosine", "euclidean"],
    similarity_matrix={
        "cosine": {
            "ref_id1": {"test_id1": 0.95, "test_id2": 0.82},
            "ref_id2": {"test_id1": 0.78, "test_id2": 0.91}
        },
        "euclidean": {
            "ref_id1": {"test_id1": 0.12, "test_id2": 0.34},
            "ref_id2": {"test_id1": 0.45, "test_id2": 0.19}
        }
    }
)
```

#### Dataset Nodes (`DataSetNode`)

Represent machine learning datasets assembled from processed features:

```python
class DataSetNode(_NodeBase):
    """Machine learning dataset node."""

    stage: Literal["dataset"] = "dataset"
    kind: Literal["dataset"] = "dataset"
```

**Purpose:**
- Mark nodes containing ML-ready datasets
- Track feature extraction sources
- Support model training workflows

#### Output Nodes (`OutputNode`)

Store final analysis results and predictions:

```python
class OutputNode(_NodeBase):
    """Final output node."""

    stage: Literal["output"] = "output"
    Fault_Type: str | None = Field(None, description="Detected fault classification")
    Confidence_Score: float | None = Field(None, description="Prediction confidence (0-1)")
    Health_Index: float | None = Field(None, description="Overall health score (0-1)")
    Anomaly_Score: float | None = Field(None, description="Anomaly detection score")
    Remaining_Life: str | None = Field(None, description="Predicted remaining useful life")
```

**Purpose:**
- Store final diagnosis results
- Provide structured output format
- Support decision making and reporting

### 2. DAG State Management (`DAGState`)

The DAGState class manages the computational graph topology and metadata:

```python
class DAGState(BaseModel):
    """DAG topology and metadata management."""

    user_instruction: str
    channels: List[str]
    nodes: Dict[str, Any] = Field(default_factory=dict)
    leaves: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
    graph_path: str | None = None
```

**Key Features:**

#### Node Management
- **Dynamic Addition**: Nodes added during execution
- **Topology Tracking**: Parent-child relationships maintained
- **Leaf Identification**: Current processing frontier tracked

#### Error Handling
- **Error Logging**: Comprehensive error tracking
- **Graceful Degradation**: Continues operation despite errors
- **Debug Information**: Detailed error context

#### Visualization Support
- **Graph Export**: JSON serialization for visualization
- **Image Generation**: PNG output for reports
- **Interactive Exploration**: Support for graph analysis tools

**Usage Example:**
```python
from src.states.phm_states import DAGState, InputData

# Initialize DAG with input channels
dag = DAGState(
    user_instruction="Bearing fault diagnosis",
    channels=["ch1", "ch2"],
    nodes={
        "ch1": InputData(node_id="ch1", parents=[], shape=(1, 1024, 2)),
        "ch2": InputData(node_id="ch2", parents=[], shape=(1, 1024, 2))
    },
    leaves=["ch1", "ch2"]
)

# Add processed node
fft_node = ProcessedData(
    node_id="fft_ch1",
    parents=["ch1"],
    source_signal_id="ch1",
    method="fft"
)
dag.nodes["fft_ch1"] = fft_node
dag.leaves = ["fft_ch1", "ch2"]  # Update leaves
```

### 3. System State (`PHMState`)

The PHMState class represents the complete system state throughout the analysis pipeline:

```python
class PHMState(BaseModel):
    """Central state for the PHM LangGraph pipeline."""

    # Core identification
    case_name: str = ""
    user_instruction: str = Field(default="", description="User's instruction")

    # Signal data
    reference_signal: InputData
    test_signal: InputData
    dag_state: DAGState

    # Processing constraints
    min_depth: int = 4
    min_width: int = 4
    max_depth: int = 8
    fs: float | None = Field(default=None, description="Sampling frequency in Hz")

    # Planning and execution
    detailed_plan: List[dict] = Field(default_factory=list)
    error_logs: List[str] = Field(default_factory=list)

    # Reflection and iteration
    reflection_history: List[str] = Field(default_factory=list)
    is_sufficient: bool = False
    iteration_count: int = 0
    needs_revision: bool = False

    # Analysis results
    insights: List[AnalysisInsight] = Field(default_factory=list)
    datasets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ml_results: Dict[str, Any] = Field(default_factory=dict)
    final_report: str = ""
```

**Key State Categories:**

#### Core Identification
- `case_name`: Unique identifier for the analysis case
- `user_instruction`: Natural language task description

#### Signal Data
- `reference_signal`: Root node for reference dataset
- `test_signal`: Root node for test dataset
- `dag_state`: Complete computational graph

#### Processing Constraints
- `min_depth`, `max_depth`: DAG complexity bounds
- `min_width`: Minimum parallel processing paths
- `fs`: Signal sampling frequency

#### Planning and Execution
- `detailed_plan`: Structured processing steps from plan_agent
- `error_logs`: System-wide error tracking

#### Reflection and Iteration
- `reflection_history`: Feedback from reflect_agent
- `is_sufficient`: Analysis completeness flag
- `iteration_count`: Processing iteration counter
- `needs_revision`: Revision requirement flag

#### Analysis Results
- `insights`: Structured analysis insights
- `datasets`: ML-ready feature datasets
- `ml_results`: Model training and evaluation results
- `final_report`: Generated markdown report

## State Initialization

### Basic Initialization

```python
from src.utils import initialize_state

# Initialize complete system state
state = initialize_state(
    user_instruction="Analyze bearing signals for fault detection",
    metadata_path="/path/to/metadata.xlsx",
    h5_path="/path/to/signals.h5",
    ref_ids=[1, 2, 3, 4, 5],  # Reference signal IDs
    test_ids=[6, 7, 8, 9, 10],  # Test signal IDs
    case_name="bearing_analysis_case1",
    use_window=True  # Apply windowing preprocessing
)
```

### Manual State Construction

```python
from src.states.phm_states import PHMState, DAGState, InputData
import numpy as np

# Create input nodes
ref_signal = InputData(
    node_id="ref_ch1",
    parents=[],
    shape=(5, 1024, 1),  # 5 reference signals
    results={"ref": reference_data},
    meta={"fs": 1000, "labels": {"id1": "healthy", "id2": "faulty"}}
)

test_signal = InputData(
    node_id="test_ch1",
    parents=[],
    shape=(3, 1024, 1),  # 3 test signals
    results={"tst": test_data},
    meta={"fs": 1000}
)

# Create DAG state
dag_state = DAGState(
    user_instruction="Bearing fault diagnosis",
    channels=["ch1"],
    nodes={"ref_ch1": ref_signal, "test_ch1": test_signal},
    leaves=["ref_ch1", "test_ch1"]
)

# Create complete system state
state = PHMState(
    case_name="manual_case",
    user_instruction="Bearing fault diagnosis",
    reference_signal=ref_signal,
    test_signal=test_signal,
    dag_state=dag_state,
    fs=1000
)
```

## State Operations

### DAG Manipulation

#### Adding Nodes

```python
from src.states.phm_states import ProcessedData

# Create new processed node
fft_node = ProcessedData(
    node_id="fft_ch1",
    parents=["ref_ch1"],
    shape=(5, 513, 1),  # FFT output shape
    source_signal_id="ref_ch1",
    method="fft",
    results={"ref": fft_results},
    meta={"tool": "fft", "params": {}}
)

# Add to DAG (immutable update)
new_nodes = state.dag_state.nodes.copy()
new_nodes["fft_ch1"] = fft_node

new_leaves = ["fft_ch1", "test_ch1"]  # Update leaves

# Create new DAG state
new_dag_state = state.dag_state.model_copy(
    update={"nodes": new_nodes, "leaves": new_leaves}
)
```

#### Querying DAG

```python
# Get all nodes of specific stage
processed_nodes = {
    node_id: node for node_id, node in state.dag_state.nodes.items()
    if getattr(node, "stage", None) == "processed"
}

# Find leaf nodes (current processing frontier)
current_leaves = state.dag_state.leaves

# Get node by ID
target_node = state.dag_state.nodes.get("fft_ch1")

# Find parent nodes
def get_parents(node_id: str) -> List[str]:
    node = state.dag_state.nodes.get(node_id)
    return node.parents if node else []

# Find children nodes
def get_children(node_id: str) -> List[str]:
    children = []
    for nid, node in state.dag_state.nodes.items():
        if node_id in node.parents:
            children.append(nid)
    return children
```

### State Persistence

#### Saving State

```python
from src.utils import save_state

# Save complete state to disk
save_state(state, "/path/to/save/state.pkl")

# Save DAG visualization
state.tracker().write_png("/path/to/dag_visualization.png")

# Export DAG as JSON
dag_json = state.tracker().export_json()
with open("/path/to/dag.json", "w") as f:
    f.write(dag_json)
```

#### Loading State

```python
from src.utils import load_state

# Load previously saved state
loaded_state = load_state("/path/to/save/state.pkl")

# Verify state integrity
assert loaded_state.case_name == state.case_name
assert len(loaded_state.dag_state.nodes) == len(state.dag_state.nodes)
```

### State Validation

#### Data Consistency Checks

```python
def validate_state(state: PHMState) -> List[str]:
    """Validate state consistency and return error list."""
    errors = []

    # Check DAG topology
    for node_id, node in state.dag_state.nodes.items():
        # Verify parent references
        for parent_id in node.parents:
            if parent_id not in state.dag_state.nodes:
                errors.append(f"Node {node_id} references missing parent {parent_id}")

    # Check leaf consistency
    for leaf_id in state.dag_state.leaves:
        if leaf_id not in state.dag_state.nodes:
            errors.append(f"Leaf {leaf_id} not found in nodes")

    # Verify signal data
    if not hasattr(state.reference_signal, "results"):
        errors.append("Reference signal missing results data")

    if not hasattr(state.test_signal, "results"):
        errors.append("Test signal missing results data")

    return errors

# Validate current state
validation_errors = validate_state(state)
if validation_errors:
    print("State validation errors:")
    for error in validation_errors:
        print(f"  - {error}")
```

## Utility Functions

### Data Access Helpers

```python
def get_node_data(state: PHMState, node_id: str) -> np.ndarray | None:
    """Utility to fetch raw array data from a node."""
    node = state.dag_state.nodes.get(node_id)
    if isinstance(node, InputData):
        return np.asarray(node.data.get("signal", []))
    elif isinstance(node, ProcessedData):
        return np.asarray(node.processed_data) if node.processed_data is not None else None
    return None

# Usage
signal_data = get_node_data(state, "ch1")
if signal_data is not None:
    print(f"Signal shape: {signal_data.shape}")
```

### DAG Analysis

```python
from src.utils import get_dag_depth

# Calculate DAG depth
current_depth = get_dag_depth(state.dag_state)
print(f"Current DAG depth: {current_depth}")

# Check if depth constraints are met
if current_depth < state.min_depth:
    print("DAG needs more processing depth")
elif current_depth > state.max_depth:
    print("DAG exceeds maximum depth")
```

### State Tracking

```python
# Access DAG tracker for advanced operations
tracker = state.tracker()

# Export graph structure
graph_json = tracker.export_json()

# Generate visualization
tracker.write_png("/path/to/output.png")

# Get networkx graph for analysis
nx_graph = tracker.to_networkx()
```

## Integration Patterns

### Agent Integration

Agents receive and modify state consistently:

```python
def example_agent(state: PHMState) -> Dict[str, Any]:
    """Example agent following state management patterns."""

    # Read current state
    current_leaves = state.dag_state.leaves
    user_instruction = state.user_instruction

    # Perform processing
    results = process_data(current_leaves)

    # Update state immutably
    new_nodes = state.dag_state.nodes.copy()
    new_nodes["new_node"] = create_new_node(results)

    # Return state updates
    return {
        "dag_state": state.dag_state.model_copy(
            update={"nodes": new_nodes, "leaves": ["new_node"]}
        ),
        "iteration_count": state.iteration_count + 1
    }
```

### LangGraph Integration

State flows through the LangGraph pipeline:

```python
from langgraph.graph import StateGraph

# Define graph with PHMState
graph = StateGraph(PHMState)

# Add nodes that operate on state
graph.add_node("plan", plan_agent)
graph.add_node("execute", execute_agent)
graph.add_node("reflect", reflect_agent_node)

# State automatically flows between nodes
graph.add_edge("plan", "execute")
graph.add_edge("execute", "reflect")
```

## Error Handling

### Error Logging

```python
# Log errors at different levels
state.error_logs.append("System-level error message")
state.dag_state.error_log.append("DAG-specific error message")

# Check for errors
if state.error_logs or state.dag_state.error_log:
    print("Errors encountered during processing:")
    for error in state.error_logs + state.dag_state.error_log:
        print(f"  - {error}")
```

### Recovery Strategies

```python
def recover_from_errors(state: PHMState) -> PHMState:
    """Implement error recovery strategies."""

    if state.dag_state.error_log:
        # Reset to last known good state
        state.needs_revision = True
        state.reflection_history.append("Errors detected, revision needed")

    if len(state.error_logs) > 10:
        # Too many errors, halt processing
        state.is_sufficient = True
        state.final_report = "Analysis halted due to excessive errors"

    return state
```

## Performance Considerations

### Memory Management

- **Immutable Updates**: State modifications create new objects to prevent side effects
- **Lazy Loading**: Large arrays loaded on demand
- **Garbage Collection**: Old state versions automatically cleaned up

### Scalability

- **Batch Processing**: Support for multiple signals in single nodes
- **Parallel Execution**: DAG structure enables parallel processing
- **Incremental Updates**: Only modified portions of state are updated

### Optimization Tips

```python
# Efficient state updates
def efficient_state_update(state: PHMState, new_node: ProcessedData) -> PHMState:
    """Efficiently update state with new node."""

    # Use model_copy for minimal object creation
    new_dag = state.dag_state.model_copy(
        update={
            "nodes": {**state.dag_state.nodes, new_node.node_id: new_node},
            "leaves": [new_node.node_id]
        }
    )

    # Return updated state
    return state.model_copy(update={"dag_state": new_dag})
```

This comprehensive state management system provides the foundation for reliable, scalable, and maintainable signal processing workflows in the PHMGA system.
```