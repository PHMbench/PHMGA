# PHMGA Cases Module

The cases module contains real-world case studies and examples demonstrating the PHMGA system's capabilities for signal processing and fault diagnosis.

## Overview

The cases module provides:
- **Complete Examples**: End-to-end analysis workflows
- **Configuration Patterns**: YAML-based configuration management
- **Best Practices**: Proven approaches for different analysis scenarios
- **Reproducible Results**: Documented case studies with expected outcomes

## Case Studies

### 1. Case 1: Standard Bearing Fault Diagnosis (`case1.py`)

#### Description

A comprehensive bearing fault diagnosis case using vibration signals from the PHM-Vibench dataset. This case demonstrates the complete PHMGA workflow from data loading to final reporting.

#### Configuration (`config/case1.yaml`)

```yaml
# Configuration for Case 1: Standard Bearing Fault Diagnosis

# File paths
name: "case1"
save_dir: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save"
metadata_path: "/mnt/crucial/LQ/PHM-Vibench/metadata_6_11.xlsx"
h5_path: "/mnt/crucial/LQ/PHM-Vibench/cache.h5"
state_save_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_built_state.pkl"
report_path: "/home/lq/LQcode/2_project/PHMBench/PHMGA/save/case1/case1_final_report.md"

# Data selection
ref_ids:
  - 47050  # Healthy bearing samples
  - 47052
  - 47044
  - 47046
  - 47047
  - 47049
  - 47053
  - 47055
  - 47056
  - 47058
test_ids:
  - 47051  # Test samples for classification
  - 47045
  - 47048
  - 47054
  - 47057

# Processing constraints
builder:
  min_depth: 4
  max_depth: 8

# User instruction for the LLM
user_instruction: >
  Analyze the bearing signals for potential faults. The reference set contains signals
  for 5 different states (health, ball, cage, inner, outer). The test set also
  contains signals for the same 5 states. The goal is to correctly classify each
  test signal by comparing it to the reference set.
```

#### Implementation

```python
def run_case(config_path: str):
    """
    Execute complete case analysis workflow.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    """
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Running Case: {config['name']} ===")
    
    # Phase 1: Build signal processing DAG
    print("Phase 1: Building signal processing DAG...")
    builder_graph = build_builder_graph()
    
    initial_state = initialize_state(
        user_instruction=config['user_instruction'],
        metadata_path=config['metadata_path'],
        h5_path=config['h5_path'],
        ref_ids=config['ref_ids'],
        test_ids=config['test_ids'],
        case_name=config['name']
    )
    
    built_state = builder_graph.invoke(initial_state)
    
    # Save intermediate state
    save_state(built_state, config['state_save_path'])
    print(f"DAG built with {len(built_state.dag_state.nodes)} nodes")
    
    # Phase 2: Execute analysis
    print("Phase 2: Executing analysis...")
    executor_graph = build_executor_graph()
    final_state = executor_graph.invoke(built_state)
    
    # Generate final report
    with open(config['report_path'], 'w') as f:
        f.write(final_state.final_report)
    
    print(f"Analysis complete. Report saved to {config['report_path']}")
    
    return final_state
```

#### Usage

```python
from src.cases.case1 import run_case

# Execute case study
final_state = run_case("config/case1.yaml")

# Examine results
print(f"Classification accuracy: {final_state.ml_results['ensemble_metrics']['accuracy']:.3f}")
print(f"F1 Score: {final_state.ml_results['ensemble_metrics']['f1']:.3f}")
```

### 2. Case 2: Advanced Signal Processing (`case_exp2.yaml`)

#### Description

Extended analysis case with advanced signal processing techniques and deeper DAG construction.

#### Key Features

- **Extended Processing**: Deeper signal processing pipelines
- **Advanced Operators**: Use of EMD, VMD, and time-frequency analysis
- **Ensemble Methods**: Multiple feature extraction paths
- **Comparative Analysis**: Cross-validation with multiple datasets

#### Configuration Highlights

```yaml
name: "exp2"
builder:
  min_depth: 6
  max_depth: 12
  
user_instruction: >
  Perform advanced bearing fault diagnosis using multiple signal processing
  techniques. Apply time-frequency analysis, empirical mode decomposition,
  and ensemble feature extraction for robust classification.
```

### 3. Case 3: Ottawa Dataset Analysis (`case_exp_ottawa.yaml`)

#### Description

Analysis of the Ottawa bearing dataset with specialized preprocessing and analysis techniques.

#### Unique Aspects

- **Dataset-Specific Preprocessing**: Tailored for Ottawa dataset characteristics
- **Multi-Channel Analysis**: Simultaneous processing of multiple sensor channels
- **Specialized Features**: Domain-specific feature extraction

## Configuration Management

### YAML Structure

All cases use standardized YAML configuration with the following sections:

```yaml
# Case identification
name: "case_name"
save_dir: "/path/to/save/directory"

# Data sources
metadata_path: "/path/to/metadata.xlsx"
h5_path: "/path/to/signals.h5"

# Output paths
state_save_path: "/path/to/saved_state.pkl"
report_path: "/path/to/final_report.md"

# Data selection
ref_ids: [list, of, reference, ids]
test_ids: [list, of, test, ids]

# Processing parameters
builder:
  min_depth: 4
  max_depth: 8
  min_width: 2

# Analysis instruction
user_instruction: "Natural language task description"

# Optional: Advanced settings
preprocessing:
  use_windowing: true
  window_size: 1024
  overlap: 0.5

analysis:
  similarity_metrics: ["cosine", "euclidean"]
  ml_algorithms: ["RandomForest", "SVM"]
  ensemble_method: "soft_voting"
```

### Configuration Loading

```python
import yaml
from pathlib import Path

def load_case_config(config_path: str) -> dict:
    """Load and validate case configuration."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    required_fields = ['name', 'metadata_path', 'h5_path', 'ref_ids', 'test_ids']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field missing: {field}")
    
    return config
```

## Case Development Guidelines

### Creating New Cases

1. **Define Objectives**: Clear analysis goals and expected outcomes
2. **Prepare Data**: Organize datasets and metadata
3. **Configure Parameters**: Set appropriate processing constraints
4. **Write Instructions**: Clear natural language task descriptions
5. **Test Execution**: Validate complete workflow
6. **Document Results**: Record expected outcomes and insights

### Case Template

```python
# src/cases/new_case.py
from __future__ import annotations
import yaml
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from src.phm_outer_graph import build_builder_graph, build_executor_graph
from src.utils import initialize_state, save_state

def run_new_case(config_path: str):
    """Execute new case analysis workflow."""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"=== Running New Case: {config['name']} ===")
    
    # Initialize state
    initial_state = initialize_state(
        user_instruction=config['user_instruction'],
        metadata_path=config['metadata_path'],
        h5_path=config['h5_path'],
        ref_ids=config['ref_ids'],
        test_ids=config['test_ids'],
        case_name=config['name']
    )
    
    # Build DAG
    builder_graph = build_builder_graph()
    built_state = builder_graph.invoke(initial_state)
    
    # Execute analysis
    executor_graph = build_executor_graph()
    final_state = executor_graph.invoke(built_state)
    
    # Save results
    save_state(final_state, config.get('state_save_path', f"{config['name']}_state.pkl"))
    
    if 'report_path' in config:
        with open(config['report_path'], 'w') as f:
            f.write(final_state.final_report)
    
    return final_state

if __name__ == "__main__":
    # Allow direct execution for testing
    run_new_case("config/new_case.yaml")
```

## Best Practices

### Data Organization

```python
# Recommended directory structure
cases/
├── config/
│   ├── case1.yaml
│   ├── case_exp2.yaml
│   └── case_exp_ottawa.yaml
├── data/
│   ├── metadata.xlsx
│   └── signals.h5
├── results/
│   ├── case1/
│   ├── exp2/
│   └── ottawa/
└── reports/
    ├── case1_report.md
    ├── exp2_report.md
    └── ottawa_report.md
```

### Error Handling

```python
def robust_case_execution(config_path: str):
    """Execute case with comprehensive error handling."""
    
    try:
        config = load_case_config(config_path)
        
        # Validate data files exist
        for path_key in ['metadata_path', 'h5_path']:
            if not Path(config[path_key]).exists():
                raise FileNotFoundError(f"Data file not found: {config[path_key]}")
        
        # Execute case
        result = run_case(config_path)
        
        print("✅ Case execution completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Case execution failed: {e}")
        # Log error details
        import traceback
        traceback.print_exc()
        return None
```

### Performance Monitoring

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description: str):
    """Context manager for timing operations."""
    start = time.time()
    print(f"Starting {description}...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Completed {description} in {elapsed:.2f} seconds")

def monitored_case_execution(config_path: str):
    """Execute case with performance monitoring."""
    
    with timer("Case execution"):
        with timer("Configuration loading"):
            config = load_case_config(config_path)
        
        with timer("State initialization"):
            initial_state = initialize_state(**config)
        
        with timer("DAG building"):
            builder_graph = build_builder_graph()
            built_state = builder_graph.invoke(initial_state)
        
        with timer("Analysis execution"):
            executor_graph = build_executor_graph()
            final_state = executor_graph.invoke(built_state)
    
    return final_state
```

## Testing Cases

### Unit Testing

```python
import pytest
from src.cases.case1 import run_case

def test_case1_execution():
    """Test case1 execution with mock data."""
    
    # Use test configuration
    test_config = "config/test_case1.yaml"
    
    result = run_case(test_config)
    
    # Validate results
    assert result is not None
    assert len(result.dag_state.nodes) > 0
    assert result.final_report != ""
    assert 'accuracy' in result.ml_results.get('ensemble_metrics', {})

def test_configuration_validation():
    """Test configuration validation."""
    
    # Test missing required field
    with pytest.raises(ValueError):
        load_case_config("config/invalid_config.yaml")
```

### Integration Testing

```python
def test_end_to_end_workflow():
    """Test complete workflow with real data."""
    
    config_path = "config/case1.yaml"
    
    # Execute complete workflow
    final_state = run_case(config_path)
    
    # Validate workflow completion
    assert final_state.is_sufficient
    assert len(final_state.reflection_history) > 0
    assert final_state.ml_results['ensemble_metrics']['accuracy'] > 0.5
```

This cases module provides a comprehensive framework for developing, executing, and managing signal processing analysis workflows in the PHMGA system.
