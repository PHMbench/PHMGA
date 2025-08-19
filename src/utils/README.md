# PHMGA Utilities Module

The utils module provides essential utility functions for the PHMGA system, including state initialization, data loading, persistence, and visualization capabilities.

## Core Functions

### 1. State Initialization

#### Primary Initialization Function

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
    """
    Initialize complete system state from data files.
    
    Parameters
    ----------
    user_instruction : str
        Natural language task description
    metadata_path : str
        Path to Excel metadata file
    h5_path : str
        Path to HDF5 signal data file
    ref_ids : list[int]
        Reference signal sample IDs
    test_ids : list[int]
        Test signal sample IDs
    case_name : str
        Unique case identifier
    use_window : bool, default=True
        Apply windowing preprocessing
    
    Returns
    -------
    PHMState
        Fully initialized system state
    """
```

#### Usage Example

```python
from src.utils import initialize_state

# Initialize state for bearing analysis
state = initialize_state(
    user_instruction="Analyze bearing signals for fault detection",
    metadata_path="/data/metadata.xlsx",
    h5_path="/data/signals.h5",
    ref_ids=[47050, 47052, 47044, 47046, 47047],
    test_ids=[47051, 47045, 47048, 47054, 47057],
    case_name="bearing_case_1",
    use_window=True
)

print(f"Initialized state with {len(state.dag_state.nodes)} nodes")
print(f"Reference signals: {state.reference_signal.shape}")
print(f"Test signals: {state.test_signal.shape}")
```

### 2. Data Loading Functions

#### Signal Data Loading

```python
def load_signal_data(
    metadata_path: str,
    h5_path: str,
    sample_ids: list[int]
) -> tuple[dict, dict, dict]:
    """
    Load signal data and metadata from files.
    
    Parameters
    ----------
    metadata_path : str
        Path to Excel metadata file
    h5_path : str
        Path to HDF5 signal data file
    sample_ids : list[int]
        Sample IDs to load
    
    Returns
    -------
    tuple[dict, dict, dict]
        (signals, labels, metadata) dictionaries
    """
```

#### HDF5 Data Access

```python
def load_signals_from_h5(h5_path: str, sample_ids: list[int]) -> tuple[list, list]:
    """
    Load signals from HDF5 file.
    
    Parameters
    ----------
    h5_path : str
        Path to HDF5 file
    sample_ids : list[int]
        Sample IDs to extract
    
    Returns
    -------
    tuple[list, list]
        (signals, labels) lists
    """
```

### 3. Data Preprocessing

#### Windowing

```python
def apply_windowing(
    signals: list[np.ndarray],
    labels: list[str],
    window_size: int = 1024,
    overlap: float = 0.5
) -> tuple[list[np.ndarray], list[str]]:
    """
    Apply sliding window preprocessing to signals.
    
    Parameters
    ----------
    signals : list[np.ndarray]
        Input signals
    labels : list[str]
        Corresponding labels
    window_size : int, default=1024
        Window size in samples
    overlap : float, default=0.5
        Overlap ratio (0-1)
    
    Returns
    -------
    tuple[list[np.ndarray], list[str]]
        Windowed signals and labels
    """
```

### 4. State Persistence

#### Saving State

```python
def save_state(state: PHMState, path: str) -> None:
    """
    Save complete state to disk.
    
    Parameters
    ----------
    state : PHMState
        State to save
    path : str
        Output file path (.pkl)
    """
```

#### Loading State

```python
def load_state(path: str) -> PHMState:
    """
    Load state from disk.
    
    Parameters
    ----------
    path : str
        Path to saved state file
    
    Returns
    -------
    PHMState
        Loaded state object
    """
```

#### Usage Example

```python
from src.utils import save_state, load_state

# Save state after processing
save_state(final_state, "analysis_results.pkl")

# Load state later
loaded_state = load_state("analysis_results.pkl")
assert loaded_state.case_name == final_state.case_name
```

### 5. DAG Analysis

#### Depth Calculation

```python
def get_dag_depth(dag_state: DAGState) -> int:
    """
    Calculate maximum depth of DAG.
    
    Parameters
    ----------
    dag_state : DAGState
        DAG to analyze
    
    Returns
    -------
    int
        Maximum depth from root to leaf
    """
```

#### Topology Analysis

```python
def analyze_dag_topology(dag_state: DAGState) -> dict:
    """
    Analyze DAG structure and properties.
    
    Parameters
    ----------
    dag_state : DAGState
        DAG to analyze
    
    Returns
    -------
    dict
        Topology metrics including depth, width, connectivity
    """
```

### 6. Visualization Support

#### DAG Visualization

```python
def visualize_dag(dag_state: DAGState, output_path: str) -> None:
    """
    Create DAG visualization.
    
    Parameters
    ----------
    dag_state : DAGState
        DAG to visualize
    output_path : str
        Output image path
    """
```

#### Feature Evolution Visualization

```python
def visualize_dag_feature_evolution_umap(
    dag_state: DAGState,
    state: PHMState,
    labels: list[str]
) -> None:
    """
    Visualize feature evolution through DAG using UMAP.
    
    Parameters
    ----------
    dag_state : DAGState
        Processing DAG
    state : PHMState
        Complete system state
    labels : list[str]
        Sample labels for coloring
    """
```

### 7. Report Generation

#### Final Report

```python
def generate_final_report(state: PHMState, output_path: str) -> None:
    """
    Generate comprehensive analysis report.
    
    Parameters
    ----------
    state : PHMState
        Complete analysis state
    output_path : str
        Output markdown file path
    """
```

## Complete Usage Examples

### Basic Workflow

```python
from src.utils import initialize_state, save_state, get_dag_depth

# 1. Initialize system
state = initialize_state(
    user_instruction="Bearing fault diagnosis",
    metadata_path="data/metadata.xlsx",
    h5_path="data/signals.h5",
    ref_ids=[1, 2, 3, 4, 5],
    test_ids=[6, 7, 8, 9, 10],
    case_name="example_analysis"
)

# 2. Check initial state
print(f"Initial DAG depth: {get_dag_depth(state.dag_state)}")
print(f"Channels: {state.dag_state.channels}")

# 3. Save initial state
save_state(state, "initial_state.pkl")
```

### Data Processing Pipeline

```python
from src.utils import load_signal_data, apply_windowing

# Load raw data
signals, labels, metadata = load_signal_data(
    "data/metadata.xlsx",
    "data/signals.h5",
    [1, 2, 3, 4, 5]
)

# Apply preprocessing
windowed_signals, windowed_labels = apply_windowing(
    signals, labels,
    window_size=1024,
    overlap=0.5
)

print(f"Original: {len(signals)} signals")
print(f"Windowed: {len(windowed_signals)} windows")
```

### Analysis and Visualization

```python
from src.utils import visualize_dag, generate_final_report

# After analysis is complete
def complete_analysis_workflow(state):
    """Complete analysis with visualization and reporting."""
    
    # Visualize final DAG
    visualize_dag(state.dag_state, "final_dag.png")
    
    # Generate comprehensive report
    generate_final_report(state, "analysis_report.md")
    
    # Save final state
    save_state(state, "final_state.pkl")
    
    print("Analysis complete!")
    print(f"DAG nodes: {len(state.dag_state.nodes)}")
    print(f"DAG depth: {get_dag_depth(state.dag_state)}")
    print(f"Report saved to: analysis_report.md")
```

## Error Handling

### Robust Data Loading

```python
def safe_initialize_state(config):
    """Initialize state with error handling."""
    
    try:
        state = initialize_state(**config)
        return state, None
    except FileNotFoundError as e:
        return None, f"Data file not found: {e}"
    except Exception as e:
        return None, f"Initialization failed: {e}"

# Usage
config = {
    "user_instruction": "Analysis task",
    "metadata_path": "data/metadata.xlsx",
    "h5_path": "data/signals.h5",
    "ref_ids": [1, 2, 3],
    "test_ids": [4, 5, 6],
    "case_name": "test"
}

state, error = safe_initialize_state(config)
if error:
    print(f"Error: {error}")
else:
    print("State initialized successfully")
```

### Data Validation

```python
def validate_data_integrity(state: PHMState) -> list[str]:
    """Validate state data integrity."""
    
    errors = []
    
    # Check signal data
    if not hasattr(state.reference_signal, 'results'):
        errors.append("Reference signal missing results")
    
    if not hasattr(state.test_signal, 'results'):
        errors.append("Test signal missing results")
    
    # Check DAG consistency
    for node_id, node in state.dag_state.nodes.items():
        for parent_id in node.parents:
            if parent_id not in state.dag_state.nodes:
                errors.append(f"Node {node_id} references missing parent {parent_id}")
    
    return errors
```

## Dependencies

### Required Packages

```python
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
h5py>=3.0.0
pickle  # Built-in
openpyxl>=3.0.0  # For Excel files

# Visualization
matplotlib>=3.5.0
networkx>=2.6.0
umap-learn>=0.5.0  # For feature evolution visualization

# Optional
seaborn>=0.11.0  # Enhanced plotting
plotly>=5.0.0    # Interactive visualizations
```

### Installation

```bash
# Install core dependencies
pip install numpy pandas h5py openpyxl matplotlib networkx

# Install optional visualization packages
pip install umap-learn seaborn plotly
```

This utilities module provides the essential infrastructure for data management, state handling, and visualization throughout the PHMGA system.
