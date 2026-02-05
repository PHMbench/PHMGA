# GEMINI.md: Project Overview for PHMGA

## Project Overview

**PHMGA (Prognostics and Health Management Graph Agent)** is a sophisticated Python-based framework for conducting Prognostics and Health Management (PHM). It automates the analysis of sensor data (e.g., vibration signals) for tasks like fault diagnosis and prediction.

The framework's core is a **dual-layer architecture**:

1.  **Outer Layer (Agentic Workflow):** A static, high-level workflow graph managed by **LangGraph**. This layer orchestrates a team of AI agents that collaborate to solve a PHM problem described in natural language. The primary agents are:
    *   **Plan Agent:** Decomposes the user's high-level instruction into a concrete, step-by-step plan.
    *   **Execute Agent:** Constructs and executes a detailed signal processing pipeline based on the plan.
    *   **Reflect Agent:** Evaluates the results and decides if the plan needs revision, enabling an iterative, self-correcting loop.
    *   **Inquirer, Dataset Preparer, and ML Agents:** Perform similarity analysis, prepare datasets, and train machine learning models.
    *   **Report Agent:** Generates a comprehensive Markdown report summarizing the entire analysis.

2.  **Inner Layer (Computational DAG):** A dynamic **Directed Acyclic Graph (DAG)** that represents the signal processing pipeline. This DAG is built by the Execute Agent and consists of a series of interconnected operators that transform and analyze the input signal data.

### Key Technologies

*   **Core Framework:** Python
*   **Agent & Graph Orchestration:** LangGraph
*   **Data Structures & Validation:** Pydantic
*   **Machine Learning:** scikit-learn, umap-learn
*   **Signal Processing:** A rich stack including `numpy`, `scipy`, `pandas`, `librosa`, `PyWavelets`, `emd`, `vmdpy`, and more.
*   **LLM Integration:** Designed to work with Gemini and OpenAI models via `langchain`.
*   **Testing:** `pytest`

## Building and Running

### 1. Installation

It is recommended to use a virtual environment.

```bash
# Clone the repository (if you haven't already)
# git clone ...
# cd PHMGA

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The framework requires API keys for the Large Language Models it uses. Create a `.env` file in the project root:

```env
# LLM Configuration (use the key for your desired provider)
GEMINI_API_KEY="your_gemini_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"

# Optional: LangSmith for debugging
LANGCHAIN_TRACING_V2="false"
```

### 3. Running an Analysis Case

The primary way to run the framework is by executing a "case" defined by a YAML configuration file.

#### Via Command Line:

The `main.py` script is the main entry point. You specify the case name, and it automatically finds the corresponding configuration file in the `config/` directory.

```bash
# Example: Run 'case1' using config/case1.yaml
python main.py case1

# You can also specify a path to a different config file
python main.py case_exp_ottawa --config config/case_exp_ottawa.yaml
```

#### Via Python Script:

You can also invoke a case directly from a Python script, as shown in the `README.md`. This is useful for integration and development.

```python
# e.g., in a file like src/cases/case1.py
from src.utils import initialize_state
from src.phm_outer_graph import build_builder_graph
import yaml

# Load configuration
with open("config/case1.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize state from the config
state = initialize_state(
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    ref_ids=config['ref_ids'],
    test_ids=config['test_ids'],
    case_name=config['name']
)

# Build the graph and invoke it
app = build_builder_graph()
result = app.invoke(state)

# The final report and other artifacts will be saved based on config
```

## Development Conventions

### Testing

The project uses `pytest` for testing. Tests are located in the `tests/` directory.

*   **Running Tests:**
    ```bash
    pytest tests/
    ```
*   **End-to-End Tests:** `tests/test_end2end.py` provides an example of how to test the full graph invocation, using a `FakeListChatModel` to mock LLM responses for deterministic testing.

### Code Style

*   Follows **PEP 8** style guidelines.
*   Uses **type hints** extensively.
*   Docstrings are written in **Google style**.

### Adding New Signal Processing Operators

The framework is designed to be extensible. You can add new signal processing capabilities by creating custom operators.

1.  **Choose a Base Class:** Select the appropriate base class from `src.tools.signal_processing_schemas` based on the operator's function:
    *   `ExpandOp`: For operators that increase data dimensionality (e.g., STFT).
    *   `TransformOp`: For operators that preserve dimensionality (e.g., filtering).
    *   `AggregateOp`: For operators that reduce dimensionality (e.g., calculating statistical features).
2.  **Implement the Operator:**
    *   Define the `op_name`, `description`, `input_spec`, and `output_spec`.
    *   Add any parameters as Pydantic `Field`s.
    *   Implement the core logic in the `execute` method.
    *   Use the `@register_op` decorator to make the operator available to the agent system.
3.  **Add Tests:** Create a new test file in the `tests/` directory to validate your new operator.

**Example from `README.md`:**
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
        # Implement the operation
        return result
```
