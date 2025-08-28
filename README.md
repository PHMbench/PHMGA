# PHMGA: Prognostics and Health Management Graph Agent Framework

[homepage](https://github.com/PHMbench/PHMGA/tree/tutorial_v1)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-Enabled-green.svg)](https://langchain-ai.github.io/langgraph/)

PHMGA is an advanced framework for Prognostics and Health Management (PHM) that combines graph-based multi-agent systems with autonomous signal processing capabilities. The framework leverages LangGraph for orchestrating complex workflows and provides intelligent agent-driven approaches for signal analysis and fault diagnosis.

## 🚀 Key Features

- **Autonomous Signal Processing**: Self-optimizing signal processing pipelines using intelligent graph agents
- **Dual-Layer Architecture**: Static outer workflow graph with dynamic inner computational DAGs
- **LLM Integration**: Natural language interfaces for algorithm configuration and optimization
- **Modular Design**: Extensible operator library for signal processing, feature extraction, and analysis
- **Multi-Agent System**: Coordinated agents for planning, execution, reflection, and reporting
- **Real-time Processing**: Support for both streaming and batch processing modes
- **Comprehensive Testing**: Built-in benchmarking and validation frameworks

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Signal Processing Operators](#signal-processing-operators)
- [Case Studies](#case-studies)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for development installation)

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/PHMbench/PHMGA.git
cd PHMGA

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install in development mode with additional tools
pip install -e .
pip install -r requirements-dev.txt  # If available

# Install pre-commit hooks
pre-commit install
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: LangSmith for debugging (set to false for cleaner logs)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=""
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT=""

# Data paths (adjust according to your setup)
PHM_DATA_PATH=/path/to/your/data
PHM_CACHE_PATH=/path/to/cache
```

## 🚀 Quick Start

### Basic Usage

```python
from src.cases.case1 import run_case

# Run a basic bearing fault diagnosis case
run_case("config/case1.yaml")
```

### Custom Configuration

```python
import yaml
from src.utils import initialize_state
from src.phm_outer_graph import build_builder_graph

# Load configuration
with open("config/case1.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize state
state = initialize_state(
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    ref_ids=config['ref_ids'],
    test_ids=config['test_ids'],
    case_name=config['name']
)

# Build and run workflow
app = build_builder_graph()
result = app.invoke(state)
```

## 🏗️ Architecture Overview

PHMGA employs a sophisticated dual-layer architecture:

### Outer Layer: Static Workflow Graph
The outer layer manages the high-level workflow using LangGraph:

```
START → Plan → Execute → Reflect → (Plan | Report) → END
```

**Core Agents:**
- **Planner Agent**: Decomposes user instructions into executable sub-goals
- **Executor Agent**: Implements dynamic "think-act" loops for signal processing
- **Reflector Agent**: Evaluates results and determines if revision is needed
- **Reporter Agent**: Generates comprehensive analysis reports

### Inner Layer: Dynamic Computational DAG
The inner layer constructs adaptive signal processing pipelines:

```
Input Signal → [Processing Operators] → Feature Extraction → Classification
```

**Operator Categories:**
- **Expand Ops**: Increase dimensionality (e.g., STFT, Mel-spectrogram)
- **Transform Ops**: Preserve dimensionality (e.g., filtering, normalization)
- **Aggregate Ops**: Reduce dimensionality (e.g., statistical features)
- **Decision Ops**: Classification and decision making

## 🧩 Core Components

### State Management (`src/states/phm_states.py`)

The `PHMState` class serves as the central data structure for the entire workflow:

```python
class PHMState(BaseModel):
    case_name: str = ""
    user_instruction: str = ""
    reference_signal: InputData
    test_signal: InputData
    dag_state: DAGState
    min_depth: int = 4
    max_depth: int = 8
    fs: float | None = None  # Sampling frequency
    detailed_plan: List[dict] = Field(default_factory=list)
    error_logs: List[str] = Field(default_factory=list)
    needs_revision: bool = False
```

### Agent System (`src/agents/`)

#### Planner Agent
- **Purpose**: Goal decomposition and high-level planning
- **Input**: User instructions and reflection history
- **Output**: Structured execution plan

#### Executor Agent
- **Purpose**: Dynamic signal processing pipeline construction
- **Features**: Think-act loops, operator selection, DAG building
- **Tools**: Access to all signal processing operators

#### Reflector Agent
- **Purpose**: Quality assessment and revision decisions
- **Capabilities**: DAG validation, performance evaluation, error detection

#### Reporter Agent
- **Purpose**: Comprehensive result analysis and documentation
- **Output**: Detailed reports with visualizations and insights

### Signal Processing Engine (`src/tools/`)

The framework provides a comprehensive library of signal processing operators:

#### Base Classes
```python
class PHMOperator(BaseModel, abc.ABC):
    """Abstract base class for all PHM operators"""
    node_id: str
    op_name: ClassVar[str]
    rank_class: ClassVar[RankClass]
    description: ClassVar[str]
    input_spec: ClassVar[str]
    output_spec: ClassVar[str]
```

#### Operator Registry
All operators are automatically registered and discoverable:

```python
from src.tools.signal_processing_schemas import get_operator, OP_REGISTRY

# Get specific operator
stft_op = get_operator("stft")

# List all available operators
available_ops = list(OP_REGISTRY.keys())
```

## 📊 Signal Processing Operators

### Expand Operators (`src/tools/expand_schemas.py`)

Transform signals to higher-dimensional representations:

| Operator | Description | Input | Output |
|----------|-------------|-------|--------|
| `patch` | Split signal into overlapping patches | `(B, L, C)` | `(B, N, P, C)` |
| `stft` | Short-Time Fourier Transform | `(B, L, C)` | `(B, F, T, C)` |
| `mel_spectrogram` | Mel-frequency spectrogram | `(B, L, C)` | `(B, M, T, C)` |
| `spectrogram` | Power spectrogram | `(B, L, C)` | `(B, F, T, C)` |
| `time_delay_embedding` | Phase space reconstruction | `(B, L, C)` | `(B, L', D, C)` |
| `vmd` | Variational Mode Decomposition | `(B, L, C)` | `(B, L, K, C)` |
| `emd` | Empirical Mode Decomposition | `(B, L, C)` | `(B, L, IMF, C)` |

### Transform Operators (`src/tools/transform_schemas.py`)

Preserve dimensionality while transforming signal characteristics:

| Operator | Description | Parameters |
|----------|-------------|------------|
| `normalize` | Signal normalization | `method`: z_score, min_max |
| `filter` | Digital filtering | `filter_type`, `cutoff`, `order` |
| `resample` | Signal resampling | `num`: new sample count |
| `psd` | Power Spectral Density | `fs`, `nperseg` |
| `power_to_db` | Convert to decibel scale | `ref`, `top_db` |

### Aggregate Operators (`src/tools/aggregate_schemas.py`)

Extract statistical and domain-specific features:

| Category | Operators | Description |
|----------|-----------|-------------|
| **Basic Stats** | `mean`, `std`, `var`, `min`, `max` | Fundamental statistics |
| **Shape Stats** | `skewness`, `kurtosis` | Distribution characteristics |
| **Signal Quality** | `snr`, `thd`, `crest_factor` | Signal integrity metrics |
| **Vibration Analysis** | `rms`, `peak_to_peak`, `clearance_factor` | Mechanical health indicators |
| **Entropy Measures** | `shannon_entropy`, `sample_entropy` | Complexity quantification |
| **Frequency Domain** | `spectral_centroid`, `spectral_rolloff` | Spectral characteristics |

## 📚 Case Studies

### Case 1: Bearing Fault Diagnosis

A comprehensive example demonstrating the framework's capabilities:

**Objective**: Classify bearing signals into health states (healthy, ball fault, cage fault, inner race fault, outer race fault)

**Configuration** (`config/case1.yaml`):
```yaml
name: "case1"
metadata_path: "/path/to/metadata.xlsx"
h5_path: "/path/to/cache.h5"
ref_ids: [47050, 47052, 47044, 47046, 47047, 47049, 47053, 47055, 47056, 47058]
test_ids: [47051, 47045, 47048, 47054, 47057]
user_instruction: >
  Analyze the bearing signals for potential faults. The reference set contains signals
  for 5 different states. The goal is to correctly classify each test signal.
```

**Execution**:
```python
from src.cases.case1 import run_case
run_case("config/case1.yaml")
```

**Expected Output**:
- Automatically constructed signal processing DAG
- Feature extraction and similarity analysis
- Classification results with confidence scores
- Comprehensive analysis report

### Case 2: Autonomous Signal Processing DAG (Coming Soon)

Advanced demonstration of self-optimizing signal processing pipelines.

## 🔌 API Reference

### Core Functions

#### State Initialization
```python
def initialize_state(
    user_instruction: str,
    metadata_path: str,
    h5_path: str,
    ref_ids: List[int],
    test_ids: List[int],
    case_name: str
) -> PHMState:
    """Initialize PHM state with data and configuration."""
```

#### Graph Construction
```python
def build_builder_graph() -> StateGraph:
    """Construct the graph for iterative DAG building."""

def build_executor_graph() -> StateGraph:
    """Construct the graph for executing finalized DAGs."""
```

#### Utility Functions
```python
def save_state(state: PHMState, path: str) -> None:
    """Save state to disk for checkpointing."""

def load_state(path: str) -> PHMState | None:
    """Load state from disk."""

def generate_final_report(state: PHMState, path: str) -> None:
    """Generate comprehensive analysis report."""
```

### Signal Processing API

#### Operator Usage
```python
from src.tools.expand_schemas import STFTOp

# Create operator instance
stft = STFTOp(fs=1000, nperseg=256, noverlap=128)

# Apply to signal
spectrogram = stft(signal_data)
```

#### Custom Operator Development
```python
from src.tools.signal_processing_schemas import register_op, TransformOp

@register_op
class CustomOp(TransformOp):
    op_name: ClassVar[str] = "custom"
    description: ClassVar[str] = "Custom signal processing operation"
    input_spec: ClassVar[str] = "(B, L, C)"
    output_spec: ClassVar[str] = "(B, L, C)"

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Implement custom logic
        return processed_signal
```

## 🤝 Contributing

We welcome contributions to the PHMGA framework! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes with appropriate tests
5. Run the test suite: `pytest tests/`
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints throughout
- Write comprehensive docstrings (Google style)
- Maintain minimum 80% test coverage
- Use meaningful variable and function names

### Adding New Operators

1. Choose the appropriate base class (`ExpandOp`, `TransformOp`, `AggregateOp`)
2. Implement the `execute` method
3. Add comprehensive tests
4. Update documentation

Example:
```python
@register_op
class NewOperator(TransformOp):
    op_name: ClassVar[str] = "new_op"
    description: ClassVar[str] = "Description of the new operator"
    input_spec: ClassVar[str] = "Input shape specification"
    output_spec: ClassVar[str] = "Output shape specification"

    # Add parameters as Pydantic fields
    param1: float = Field(..., description="Parameter description")

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Implement the operation
        return result
```

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: `ImportError: No module named 'langgraph'`
**Solution**:
```bash
pip install --upgrade langgraph langgraph-prebuilt
```

**Issue**: `ModuleNotFoundError: No module named 'vmdpy'`
**Solution**:
```bash
pip install vmdpy
```

#### Runtime Errors

**Issue**: `ValueError: Input for STFTOp must be 3D (B, L, C), but got 2D`
**Solution**: Ensure input signals have the correct shape. Reshape if necessary:
```python
if signal.ndim == 2:
    signal = signal.reshape(1, -1, 1)  # Add batch and channel dimensions
```

**Issue**: `KeyError: 'op_name' not found in OP_REGISTRY`
**Solution**: Verify the operator is properly registered:
```python
from src.tools.signal_processing_schemas import OP_REGISTRY
print(list(OP_REGISTRY.keys()))  # List available operators
```

#### Performance Issues

**Issue**: Slow execution with large signals
**Solution**:
- Use appropriate patch sizes for time-frequency transforms
- Consider downsampling for exploratory analysis
- Enable parallel processing where available

**Issue**: Memory errors with large datasets
**Solution**:
- Process signals in batches
- Use memory-efficient operators
- Monitor memory usage with profiling tools

### Debugging Tips

1. **Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Use state checkpointing**:
```python
from src.utils import save_state, load_state
save_state(current_state, "debug_checkpoint.pkl")
```

3. **Visualize DAG structure**:
```python
from src.utils.visualization import visualize_dag
visualize_dag(state.dag_state)
```

### Getting Help

- **Documentation**: Check the tutorial series in `/tutorials/`
- **Issues**: Report bugs on [GitHub Issues](https://github.com/PHMbench/PHMGA/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/PHMbench/PHMGA/discussions)
- **Email**: Contact the maintainers at [email@example.com]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- LangGraph team for the excellent workflow orchestration framework
- The PHM research community for domain expertise and validation
- Contributors and early adopters for feedback and improvements

## 📈 Roadmap

- [ ] Enhanced LLM integration with more model providers
- [ ] Real-time streaming processing capabilities
- [ ] Web-based dashboard for interactive analysis
- [ ] Extended operator library with deep learning components
- [ ] Multi-modal signal processing (audio, vibration, thermal)
- [ ] Distributed processing for large-scale deployments

---

**Note**: This framework is under active development. Please check the [CHANGELOG](CHANGELOG.md) for recent updates and breaking changes.
