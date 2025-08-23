# PHMGA State Management System Documentation

This document provides comprehensive guidance for working with the PHMGA state management system.

## Architecture Overview

The state management system consists of three main layers:

1. **Node Layer**: Individual data nodes representing signals and processing results
2. **DAG Layer**: Graph topology and relationships between nodes
3. **System Layer**: Overall system state including configuration and metadata

## PHMState (Central State Structure)

```python
class PHMState(BaseModel):
    # Core identification
    case_name: str = ""
    user_instruction: str = ""
    
    # Signal data
    reference_signal: InputData
    test_signal: InputData  
    dag_state: DAGState
    
    # Configuration
    llm_provider: str = "google"  # or "openai"
    llm_model: str = "gemini-2.5-pro"
    min_depth: int = 4
    max_depth: int = 8
    fs: Optional[float] = None
    
    # Processing state
    detailed_plan: List[dict] = Field(default_factory=list)
    error_logs: List[str] = Field(default_factory=list)
    reflection_history: List[str] = Field(default_factory=list)
    needs_revision: bool = False
    
    # Results
    insights: List[AnalysisInsight] = Field(default_factory=list)
    datasets: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ml_results: Dict[str, Any] = Field(default_factory=dict)
    final_report: str = ""
```

## Node Types

### Base Node Class (`_NodeBase`)

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

### InputData Nodes

```python
class InputData(_NodeBase):
    """Raw input signal node."""
    stage: str = "input"
    data: Dict[str, Any] = Field(default_factory=dict)
    results: Dict[str, Any] = Field(default_factory=dict)  # "ref", "tst"
    meta: Dict[str, Any] = Field(default_factory=dict)     # fs, labels
```

**Usage:**
```python
ch1_node = InputData(
    node_id="ch1",
    parents=[],
    shape=(1, 1024, 2),
    results={"ref": reference_signals, "tst": test_signals},
    meta={"fs": 1000, "labels": {"id1": "healthy", "id2": "faulty"}}
)
```

### ProcessedData Nodes

```python
class ProcessedData(_NodeBase):
    """Processed signal node."""
    stage: str = "processed"
    source_signal_id: str
    method: str
    results: Dict[str, np.ndarray] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
```

### Other Node Types

- **SimilarityNode**: Stores similarity analysis results between signals
- **DataSetNode**: ML-ready datasets with features and labels
- **OutputNode**: Final classification results with fault type, confidence, health index

## DAGState Management

```python
class DAGState(BaseModel):
    """DAG topology and metadata management."""
    user_instruction: str
    channels: List[str]
    nodes: Dict[str, Any] = Field(default_factory=dict)
    leaves: List[str] = Field(default_factory=list)
    error_log: List[str] = Field(default_factory=list)
```

**Key Operations:**
- Add nodes immutably with topology updates
- Track leaf nodes (current processing frontier)
- Validate parent-child relationships
- Export to JSON/PNG for visualization

## State Operations

### State Initialization

```python
from src.utils import initialize_state

state = initialize_state(
    user_instruction="Bearing fault diagnosis",
    metadata_path="/data/metadata.xlsx", 
    h5_path="/data/signals.h5",
    ref_ids=[47050, 47052, 47044],
    test_ids=[47051, 47045, 47048],
    case_name="bearing_case_1",
    use_window=True
)
```

### Immutable Updates

```python
# Add new node to DAG
new_nodes = state.dag_state.nodes.copy()
new_nodes["fft_ch1"] = fft_node

# Update state immutably
new_dag_state = state.dag_state.model_copy(
    update={"nodes": new_nodes, "leaves": ["fft_ch1"]}
)

new_state = state.model_copy(update={"dag_state": new_dag_state})
```

### State Persistence

```python
from src.utils import save_state, load_state

# Save complete state
save_state(final_state, "analysis_results.pkl")

# Load state later
loaded_state = load_state("analysis_results.pkl")
```

## Validation and Error Handling

```python
def validate_state(state: PHMState) -> List[str]:
    """Validate state consistency and return error list."""
    errors = []

    # Check DAG topology
    for node_id, node in state.dag_state.nodes.items():
        for parent_id in node.parents:
            if parent_id not in state.dag_state.nodes:
                errors.append(f"Node {node_id} references missing parent {parent_id}")

    return errors
```

## Integration Patterns

### Agent Integration

```python
def example_agent(state: PHMState) -> Dict[str, Any]:
    """Example agent following state management patterns."""
    
    # Read current state
    current_leaves = state.dag_state.leaves
    user_instruction = state.user_instruction
    
    # Perform processing
    results = process_data(current_leaves)
    
    # Return immutable state updates
    return {
        "dag_state": state.dag_state.model_copy(
            update={"nodes": new_nodes, "leaves": ["new_node"]}
        ),
        "iteration_count": state.iteration_count + 1
    }
```

This state management system provides the foundation for reliable, scalable, and maintainable signal processing workflows in the PHMGA system.