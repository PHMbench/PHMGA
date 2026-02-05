# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PHMGA (Prognostics and Health Management Graph Agent)** is a Python framework for automated sensor data analysis (e.g., vibration signals for fault diagnosis). It uses a dual-layer architecture:

1. **Outer Layer (LangGraph orchestration):** Static workflow graph with AI agents (Plan, Execute, Reflect, Inquirer, Dataset Preparer, ML, Report) that collaborate to solve PHM problems described in natural language.
2. **Inner Layer (Computational DAG):** Dynamic directed acyclic graph representing the signal processing pipeline, built by the Execute Agent.

**Key Technologies:** LangGraph, Pydantic, scikit-learn, librosa, PyWavelets, PyEMD, vmdpy, ruptures, umap-learn

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"
LANGCHAIN_TRACING_V2="false"
```

## Running Cases

Cases are defined by YAML configuration files in `config/`.

```bash
# Run a case (loads config/case1.yaml by default)
python main.py case1

# Specify a custom config path
python main.py case_exp_ottawa --config config/case_exp_ottawa.yaml
```

The `main.py` entry point dynamically imports `src/cases/<case_name>.py` and calls its `run_case(config_path)` function.

## Code Structure

- **[main.py](main.py)** - CLI entry point that loads and runs cases
- **[src/phm_outer_graph.py](src/phm_outer_graph.py)** - Defines `build_builder_graph()` (Plan→Execute→Reflect loop) and `build_executor_graph()` (Inquire→Prepare→Train→Report)
- **[src/agents/](src/agents/)** - Agent implementations (plan_agent, execute_agent, reflect_agent, inquirer_agent, dataset_preparer_agent, shallow_ml_agent, report_agent)
- **[src/cases/](src/cases/)** - Individual case implementations, each with a `run_case()` function
- **[src/tools/](src/tools/)** - Signal processing operators (transform, aggregation, decision tools)
- **[src/schemas/](src/schemas/)** - Pydantic models for data validation
- **[src/states/](src/states/)** - State management (PHMState for workflow state, DAGState for computational graph)
- **[config/](config/)** - YAML configuration files defining data paths, user instructions, and builder parameters

## Configuration File Format

See [config/case1.yaml](config/case1.yaml) for an example:

```yaml
name: "case1"
metadata_path: "path/to/metadata.xlsx"
h5_path: "path/to/cache.h5"
state_save_path: "path/to/save/built_state.pkl"
report_path: "path/to/save/report.md"

ref_ids: [47050, 47052, ...]  # Reference dataset IDs
test_ids: [47051, 47045, ...]  # Test dataset IDs

builder:
  min_depth: 4
  max_depth: 8

user_instruction: >
  Analyze the bearing signals for potential faults...
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_end2end.py
```

End-to-end tests use `FakeListChatModel` to mock LLM responses for deterministic testing.

## Adding New Signal Processing Operators

1. Choose a base class from `src.tools.signal_processing_schemas`:
   - `ExpandOp` - Increases data dimensionality (e.g., STFT)
   - `TransformOp` - Preserves dimensionality (e.g., filtering)
   - `AggregateOp` - Reduces dimensionality (e.g., statistical features)

2. Implement the operator:

```python
from src.tools.signal_processing_schemas import register_op, TransformOp

@register_op
class NewOperator(TransformOp):
    op_name: ClassVar[str] = "new_op"
    description: ClassVar[str] = "Description of the new operator"
    input_spec: ClassVar[str] = "Input shape specification"
    output_spec: ClassVar[str] = "Output shape specification"

    param1: float = Field(..., description="Parameter description")

    def execute(self, x: np.ndarray, **kwargs) -> np.ndarray:
        # Implement operation
        return result
```

3. Add tests in `tests/`.

## Running a Case Programmatically

```python
from src.utils import initialize_state
from src.phm_outer_graph import build_builder_graph
import yaml

with open("config/case1.yaml", "r") as f:
    config = yaml.safe_load(f)

state = initialize_state(
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    ref_ids=config['ref_ids'],
    test_ids=config['test_ids'],
    case_name=config['name']
)

app = build_builder_graph()
result = app.invoke(state)
```

## State Persistence

Built DAG states can be saved and loaded to skip the builder workflow:

- `save_state(state, path)` - Saves state to pickle file
- `load_state(path)` - Loads state from pickle file

The builder workflow checks for `state_save_path` and skips building if a valid state file exists.
