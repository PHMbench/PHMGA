# PHMGA Utilities Documentation

This document provides comprehensive guidance for working with the PHMGA utility functions.

## Core Functions

### State Initialization

```python
def initialize_state(
    user_instruction: str,
    metadata_path: str,
    h5_path: str,
    ref_ids: list[int],
    test_ids: list[int],
    case_name: str,
    use_window: bool = True
) -> PHMState:
    """Initialize complete system state from data files."""
```

**Usage Example:**
```python
from src.utils import initialize_state

state = initialize_state(
    user_instruction="Analyze bearing signals for fault detection",
    metadata_path="/data/metadata.xlsx",
    h5_path="/data/signals.h5",
    ref_ids=[47050, 47052, 47044],
    test_ids=[47051, 47045, 47048],
    case_name="bearing_case_1",
    use_window=True
)
```

### Data Loading

```python
from src.utils import load_signal_data, apply_windowing

# Load raw data
signals, labels, metadata = load_signal_data(metadata_path, h5_path, sample_ids)

# Apply preprocessing  
windowed_signals, windowed_labels = apply_windowing(
    signals, labels, window_size=1024, overlap=0.5
)
```

### State Persistence

```python
from src.utils import save_state, load_state

# Save complete state
save_state(final_state, "analysis_results.pkl")

# Load state later
loaded_state = load_state("analysis_results.pkl")
```

### DAG Analysis

```python
from src.utils import get_dag_depth, visualize_dag

# Calculate DAG depth
depth = get_dag_depth(state.dag_state)

# Generate visualization
visualize_dag(state.dag_state, "dag_visualization.png")
```

### Report Generation

```python
from src.utils import generate_final_report

generate_final_report(state, "comprehensive_report.md")
```

## Visualization Support

### DAG Visualization

```python
def visualize_dag(dag_state: DAGState, output_path: str) -> None:
    """Create DAG visualization."""
```

### Feature Evolution Visualization

```python
def visualize_dag_feature_evolution_umap(
    dag_state: DAGState,
    state: PHMState,
    labels: list[str]
) -> None:
    """Visualize feature evolution through DAG using UMAP."""
```

## Error Handling

```python
def validate_data_integrity(state: PHMState) -> list[str]:
    """Validate state data integrity."""
    
    errors = []
    
    # Check signal data
    if not hasattr(state.reference_signal, 'results'):
        errors.append("Reference signal missing results")
    
    # Check DAG consistency
    for node_id, node in state.dag_state.nodes.items():
        for parent_id in node.parents:
            if parent_id not in state.dag_state.nodes:
                errors.append(f"Node {node_id} references missing parent {parent_id}")
    
    return errors
```

This utilities module provides the essential infrastructure for data management, state handling, and visualization throughout the PHMGA system.