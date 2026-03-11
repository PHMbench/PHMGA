# PHMGA Agents

This directory contains the workflow agents that orchestrate the **PHMGA (Prognostics and Health Management Graph Agent)** framework. In this repository, `agent` means an outer-workflow node with a stable I/O contract. Some agents are LLM-mediated, others are deterministic.

## Architecture Overview

PHMGA uses a **dual-layer architecture**:

1. **Outer Layer (Workflow Orchestration)**: LangGraph-based workflow with agents that plan, execute, reflect, report, train, and research
2. **Inner Layer (Computational DAG)**: Dynamic directed acyclic graph representing the signal processing pipeline, built by the Execute Agent

```
┌─────────────────────────────────────────────────────────────────┐
│                         PHMGA Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  Outer Layer: LangGraph Orchestration (Workflow Agents)         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐       │
│  │  Plan   │ -> │ Execute │ -> │ Reflect │ -> │ Report  │       │
│  │  Agent  │    │  Agent  │    │  Agent  │    │  Agent  │       │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘       │
│        │              │                                              │
│        └──────────────┘                                              │
│                       │                                             │
│  Inner Layer: Computational DAG (Signal Processing)               │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───────┐     │
│  │FFT│ -> │PSD│ -> │WF │ -> │AGG│ -> │...│ -> │ML │ -> │Report │     │
│  └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Agent Summary

| Agent | File | Type | Purpose | Workflow |
|-------|------|------|---------|----------|
| **Plan Agent** | [plan_agent.py](plan_agent.py) | LLM-mediated | Generates detailed processing plan | Builder |
| **Execute Agent** | [execute_agent.py](execute_agent.py) | Hybrid | Executes DAG steps and can route to TSPN training | Builder |
| **Reflect Agent** | [reflect_agent.py](reflect_agent.py) | LLM-mediated | Evaluates DAG quality and decides next actions | Builder |
| **Inquirer Agent** | [inquirer_agent.py](inquirer_agent.py) | Deterministic | Calculates similarity metrics between signals | Executor |
| **Dataset Preparer Agent** | [dataset_preparer_agent.py](dataset_preparer_agent.py) | Deterministic | Assembles training datasets from DAG nodes | Executor |
| **Shallow ML Agent** | [shallow_ml_agent.py](shallow_ml_agent.py) | Deterministic | Trains traditional ML models (RF, SVM) | Executor |
| **Deep Model Train Agent** | [deep_model_train_agent.py](deep_model_train_agent.py) | Deterministic | Trains TSPN deep learning model | Executor |
| **TSPN Bootstrap Agent** | [tspn_bootstrap_agent.py](tspn_bootstrap_agent.py) | Deterministic | Auto-generates TSPN config from DAG | Executor |
| **DAG Init Agent** | [dag_init_agent.py](dag_init_agent.py) | LLM-mediated | Initializes minimal processed DAG | Executor |
| **Report Agent** | [report_agent.py](report_agent.py) | LLM-mediated with template fallback | Generates final markdown report | Executor |
| **Deep Research Agents** | [deep_research_agents.py](deep_research_agents.py) | LLM-mediated research subgraph | Web research with query generation | Separate |

## Classification

- `LLM-mediated agent`: the node depends on an LLM to make the core business decision.
- `Deterministic agent`: the node is still a formal workflow step, but its logic is data/config driven rather than prompt driven.
- `agent` is not a synonym for “uses an LLM”. It is the unit of orchestration in the outer workflow.
- The flat files under `src/agents/*.py` are still the formal implementations. Subdirectories such as `builder/`, `executor/`, `report/`, and `train/` are mostly compatibility shells today.

---

## Workflows

### Builder Graph (DAG Construction)

The Builder workflow iteratively constructs a computational DAG through a plan-execute-reflect loop:

```
┌────────────────────────────────────────────────────────────┐
│                    Builder Workflow                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│   │   Plan  │ --> │ Execute │ --> │ Reflect │             │
│   │  Agent  │     │  Agent  │     │  Agent  │             │
│   └─────────┘     └─────────┘     └────┬────┘             │
│       │                                 │                   │
│       └─────────────────────────────────┘                   │
│                        │                                    │
│                needs_revision?                              │
│                   /    \                                     │
│               Yes       No                                   │
│               /          \                                   │
│          (loop)        (finish)                              │
└────────────────────────────────────────────────────────────┘
```

**Entry Point**: `build_builder_graph()` in [phm_outer_graph.py](../phm_outer_graph.py#L89)

**Termination Conditions**:
- `needs_revision = False` (Reflect agent decides DAG is sufficient)
- `max_depth` reached

### Executor Graph (Analysis & Training)

The Executor workflow runs the finalized DAG for analysis and reporting:

```
┌────────────────────────────────────────────────────────────┐
│                   Executor Workflow                         │
├────────────────────────────────────────────────────────────┤
│                                                             │
│                       ┌──────┐                             │
│                       │ route│                             │
│                       └──┬───┘                             │
│                          │                                 │
│              train_backend? │                               │
│              ┌──────────────┴──────────────┐               │
│              │                             │               │
│         tspn/tspn+both               shallow/both           │
│              │                             │               │
│      ┌───────▼────────┐         ┌─────────▼────────┐      │
│      │ tspn_fast_path │         │   full_path      │      │
│      └───────┬────────┘         └─────────┬────────┘      │
│              │                             │               │
│      ┌───────▼────────┐         ┌─────────▼────────┐      │
│      │    init_dag    │         │    inquire       │      │
│      │    (optional)  │         └─────────┬────────┘      │
│      └───────┬────────┘                   │               │
│              │                ┌───────────▼────────┐      │
│      ┌───────▼────────┐       │     prepare       │      │
│      │   bootstrap    │       └───────────┬────────┘      │
│      └───────┬────────┘                   │               │
│              │                ┌───────────▼────────┐      │
│      ┌───────▼────────┐       │     init_dag       │      │
│      │     train      │       │     (optional)     │      │
│      └───────┬────────┘       └───────────┬────────┘      │
│              │                             │               │
│      ┌───────▼────────┐       ┌───────────▼────────┐      │
│      │    report      │ <─────│    bootstrap       │      │
│      └────────────────┘       └───────────┬────────┘      │
│                                         │                  │
│                                 ┌───────▼────────┐         │
│                                 │     train      │         │
│                                 └───────┬────────┘         │
│                                         │                  │
│                                 ┌───────▼────────┐         │
│                                 │    report      │         │
│                                 └────────────────┘         │
└────────────────────────────────────────────────────────────┘
```

**Entry Point**: `build_executor_graph()` in [phm_outer_graph.py](../phm_outer_graph.py#L131)

**Path Selection**:
- `tspn_fast_path`: Direct route when `train_backend="tspn"` (skips similarity analysis)
- `full_path`: Full analysis with similarity metrics when `train_backend="shallow"` or `"both"`

---

## Agent Details

### Builder Workflow Agents

#### Plan Agent

**File**: [plan_agent.py](plan_agent.py)

**Purpose**: Generates a detailed processing plan using LLM with structured output. Analyzes the current DAG state and proposes the next set of signal processing operations.

**Function**: `plan_agent(state: PHMState) -> dict`

**Input State**:
- `user_instruction`: Original user request
- `dag_state`: Current DAG topology
- `reflection_history`: Previous reflection results
- `min_depth`, `max_depth`: DAG size constraints
- `fs`: Sampling frequency (auto-injected)

**Output State**:
- `detailed_plan`: List of processing steps, each with `parent`, `op_name`, `params`

**Key Features**:
- Uses lightweight JSON-based structured output (not `.with_structured_output()`)
- Auto-injects sampling frequency `fs` for operators that require it
- Creates topology-only DAG representation for LLM context

**Dependencies**:
- LLM (Gemini/OpenAI)
- `OP_REGISTRY`: Available signal processing operators

---

#### Execute Agent

**File**: [execute_agent.py](execute_agent.py)

**Purpose**: Executes the detailed processing plan, handling both signal processing DAG operations and neuro-symbolic (TSPN) training.

**Function**: `execute_agent(state: PHMState) -> Dict[str, Any]`

**Input State**:
- `detailed_plan`: Processing steps from Plan Agent
- `dag_state`: Current DAG nodes and leaves
- `task_type`: "signal_processing_dag" or "neuro_symbolic_train"

**Output State**:
- `dag_state`: Updated DAG with new nodes
- `executed_steps`: Number of successfully executed steps

**Key Internal Functions**:
- `_resolve_params()`: Auto-generates missing operator parameters via LLM
- `_execute_single_variable_op()`: Handles operators with one parent
- `_execute_multi_variable_op()`: Handles operators with multiple parents (e.g., cross-correlation)

**Key Features**:
- Immutable DAG updates (creates new state objects)
- Saves intermediate results as `.npy`/`.npz` files
- Exports DAG visualization (PNG/DOT) after each iteration
- Supports two modes: signal processing DAG execution and neuro-symbolic training

**Dependencies**:
- All signal processing operators from `src/tools/`
- TSPN training (for neuro-symbolic mode)
- `PHMVibenchDataFactory` for vibench backend

---

#### Reflect Agent

**File**: [reflect_agent.py](reflect_agent.py)

**Purpose**: Quality checks the DAG and returns a decision with reason. Always calls LLM for intelligent reflection rather than using hardcoded rules.

**Function**: `reflect_agent_node(state: PHMState, *, stage: str) -> Dict[str, Any]`

**Input State**:
- `user_instruction`: Original task
- `dag_state`: Current DAG structure
- `min_depth`, `max_depth`: Depth constraints
- `error_log`: Execution issues (if any)

**Output State**:
- `needs_revision`: Boolean controlling builder loop
- `reflection_history`: Appended with reflection reason

**Decisions**:
- `finish`: DAG is sufficient, exit builder loop
- `need_patch`: Minor fixes needed
- `need_replan`: Major restructuring needed
- `halt`: Critical error

**Key Features**:
- Always LLM-driven for flexible decision-making
- Considers DAG depth, error logs, and user instruction
- Debug mode via `PHM_DEBUG_REFLECT` environment variable

---

### Executor Workflow Agents

#### Inquirer Agent

**File**: [inquirer_agent.py](inquirer_agent.py)

**Purpose**: Performs similarity analysis between reference and test signals using multiple metrics.

**Function**: `inquirer_agent(state: PHMState, metrics: List[str]) -> Dict[str, List[str]]`

**Input State**:
- `dag_state.leaves`: Leaf nodes containing processed data

**Output State**:
- `new_nodes`: IDs of created similarity nodes (for paired ref/tst layout)
- OR in-place `node.sim` attributes (for legacy single-node layout)

**Supported Metrics**:
- `cosine`: Cosine similarity
- `euclidean`: Euclidean distance
- `pearson`: Pearson correlation distance

**Key Features**:
- Supports two data layouts for backward compatibility
- Creates new similarity nodes with stage="similarity"

---

#### Dataset Preparer Agent

**File**: [dataset_preparer_agent.py](dataset_preparer_agent.py)

**Purpose**: Gathers features from processed nodes and assembles datasets with true labels obtained by traversing DAG to root nodes.

**Function**: `dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> Dict`

**Input State**:
- `dag_state.nodes`: ProcessedData nodes with saved features
- Node metadata with `labels_train`, `labels_val`, and `labels_test`

**Output State**:
- `datasets`: Dictionary mapping node_id to `{X_train, X_val, X_test, y_train, y_val, y_test}`
- `n_nodes`: Number of datasets created

**Key Features**:
- Traverses DAG to find labels at root (backward-compatible)
- Enforces label boundary: `labels_train` for training, `labels_test` for reporting only
- Creates `DataSetNode` additions to DAG

---

#### Shallow ML Agent

**File**: [shallow_ml_agent.py](shallow_ml_agent.py)

**Purpose**: Trains traditional ML models (Random Forest, SVM) per dataset with ensemble inference.

**Function**: `shallow_ml_agent(datasets: Dict, *, algorithm: str, ensemble_method: str, cv_folds: int) -> Dict[str, Any]`

**Input**:
- `datasets`: Dictionary of `{X_train, y_train, X_test, y_test}` arrays

**Output**:
- `models`: Trained models with metrics
- `ensemble_metrics`: Ensemble accuracy and F1
- `metrics_markdown`: Formatted results table

**Key Features**:
- Cross-validation with configurable folds
- Ensemble voting using only high-quality models (CV accuracy > 90%)
- Supports hard and soft voting
- Models encoded as base64 for portability

---

#### Deep Model Train Agent

**File**: [deep_model_train_agent.py](deep_model_train_agent.py)

**Purpose**: Inner-loop trainer for the TSPN (Time Series Predictive Network) deep learning model.

**Function**: `deep_model_train_agent(state: PHMState, *, config: Dict | None = None) -> Dict[str, Any]`

**Input State**:
- `dag_state`: DAG with channel roots
- `data_cfg`: Backend configuration ("vibench" or default)
- `model_config_path`: Resolved config written into `PHMState` by the runner
- `labels_train`, `labels_val`, `labels_test`: Canonical label mappings

**Output State**:
- `ml_results`: Training metrics and artifacts
- `run_dir`: Path to saved artifacts
- `train_history`: Updated with `TrainReport`
- `model_config_path`: Path to config YAML

**Key Features**:
- Enforces label boundary: train/val only for fitting, test only for reporting
- Supports two backends: PHM-Vibench factory and direct array loading
- Generates explainability artifacts (operator importance, wavefilters)
- Outputs predictions CSV and confusion matrix

**Dependencies**:
- PyTorch (optional - graceful failure if unavailable)
- `src.model.explainable`: TSPN model builder

---

#### TSPN Bootstrap Agent

**File**: [tspn_bootstrap_agent.py](tspn_bootstrap_agent.py)

**Purpose**: Deterministically bootstraps a minimal, valid TSPN `model_config.yaml` from a built DAG without using LLM. Reduces hand-designed priors by inferring structure from the DAG.

**Function**: `tspn_bootstrap_agent(state: PHMState, *, max_layers, parallel_ops_per_layer, out_channels, scale, features) -> Dict[str, Any]`

**Input State**:
- `dag_state`: Built DAG with ProcessedData nodes
- `labels_train`: For inferring num_classes

**Output State**:
- `model_config_path`: Path to generated YAML
- `current_model_config`: Config dictionary

**Key Features**:
- No-LLM bootstrap (deterministic)
- Infers `in_dim`, `in_channels`, `num_classes` from DAG
- Maps DAG methods (fft, hilbert, wavefilter) to TSPN tokens
- Ensures divisibility constraints

---

#### DAG Init Agent

**File**: [dag_init_agent.py](dag_init_agent.py)

**Purpose**: Lightweight bridge between outer loop (LLM) and inner loop (TSPN). Initializes minimal processed DAG with preprocessing nodes.

**Function**: `dag_init_agent(state: PHMState, *, max_ops_per_channel: int, temperature: float) -> Dict[str, Any]`

**Input State**:
- `dag_state`: DAG with only channel roots (InputData nodes)

**Output State**:
- `dag_state`: Updated with ProcessedData nodes

**Key Features**:
- Creates per-channel preprocessing chains (fft, hilbert, wavefilter)
- Uses LLM for op selection with fallback to FFT
- Maintains (1, L, 1) shape contract for bootstrap compatibility

---

#### Report Agent

**File**: [report_agent.py](report_agent.py)

**Purpose**: Generates final markdown report via LLM, with template fallback.

**Function**: `report_agent_node(state: PHMState) -> Dict[str, str]`

**Input State**:
- `user_instruction`: Original task
- `dag_state`: Final DAG structure
- `ml_results`: Training results
- Similarity stats from leaf nodes

**Output State**:
- `final_report`: Markdown report

**Key Features**:
- Multiple modes: auto, template (via `PHM_REPORT_MODE`)
- Exports final DAG visualization
- Template fallback when LLM unavailable

---

### Separate Workflow: Deep Research Agents

**File**: [deep_research_agents.py](deep_research_agents.py)

**Purpose**: Web research capability with query generation, searching, and reflection. Separate from the main PHMGA workflow.

**Nodes**:
- `generate_query`: Generates search queries
- `web_research`: Performs web searches with Google Search API
- `reflection`: Identifies knowledge gaps
- `evaluate_research`: Decides to continue or finalize
- `finalize_answer`: Creates final report with citations

**Key Features**:
- Multi-loop research with structured output
- Citation tracking and URL resolution
- Configurable max research loops

---

## State Management

All agents communicate through the shared `PHMState` object defined in [src/states/phm_states.py](../states/phm_states.py). Key state fields:

### Core DAG Fields
- `reference_signal`: InputData - Legacy root anchor kept for compatibility
- `test_signal`: InputData - Legacy root anchor kept for compatibility
- `dag_state`: DAGState - Complete DAG with nodes and leaves
- `leaves`: List[str] - Current leaf node IDs

### Execution Control
- `user_instruction`: str - Original user request
- `needs_revision`: bool - Controls builder loop
- `iteration_count`: int - Number of builder iterations
- `detailed_plan`: List[dict] - Current execution plan

### ML Configuration
- `train_backend`: str - "shallow" | "tspn" | "both"
- `datasets`: Dict[str, Any] - Prepared training datasets
- `ml_results`: Dict[str, Any] - Training results
- `model_config_path`: str | None - TSPN config path

### Configuration
- `data_cfg`: Dict - Backend configuration
- `save_dir`: str - Base save directory
- `case_name`: str - Case identifier

Runtime split semantics are `train/val/test`. `reference_signal/test_signal` remain only as historical root anchors.

---

## Usage Examples

### Running the Builder Workflow

```python
from src.utils import initialize_state
from src.phm_outer_graph import build_builder_graph
import yaml

# Load configuration
with open("config/case1.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize state
state = initialize_state(
    user_instruction=config['user_instruction'],
    metadata_path=config['metadata_path'],
    h5_path=config['h5_path'],
    train_ids=config['data']['selection']['train_ids'],
    val_ids=config['data']['selection'].get('val_ids', []),
    test_ids=config['data']['selection']['test_ids'],
    case_name=config['name']
)

# Run builder workflow
app = build_builder_graph()
for chunk in app.stream(state):
    for node_name, update in chunk.items():
        print(f"Node {node_name}: {update.keys()}")
```

### Running the Executor Workflow

```python
from src.phm_outer_graph import build_executor_graph

# Assuming state contains a built DAG from the builder
state.train_backend = "tspn"  # or "shallow" or "both"
state.allow_test_labels_for_reporting = False

executor_app = build_executor_graph()
for chunk in executor_app.stream(state):
    for node_name, update in chunk.items():
        print(f"Node {node_name}: {update.keys()}")

# Access final report
print(state.final_report)
```

### Using Individual Agents

```python
from src.agents.plan_agent import plan_agent
from src.agents.execute_agent import execute_agent
from src.agents.reflect_agent import reflect_agent_node

# Plan step
plan_result = plan_agent(state)
state.detailed_plan = plan_result["detailed_plan"]

# Execute step
exec_result = execute_agent(state)
state.dag_state = exec_result["dag_state"]

# Reflect step
reflect_result = reflect_agent_node(state, stage="POST_EXECUTE")
state.needs_revision = reflect_result["needs_revision"]
```

---

## Agent Development Guidelines

When adding new agents to the PHMGA framework:

### 1. Function Signature Pattern

Most agents follow this pattern:

```python
from src.states.phm_states import PHMState

def my_agent(state: PHMState, *, config_param: str = "default") -> Dict[str, Any]:
    """
    Purpose: Brief description of what this agent does.

    Args:
        state: Current PHMState
        config_param: Optional configuration parameter

    Returns:
        Dictionary with state updates (e.g., {"new_field": value})
    """
    # Agent logic here
    return {"new_field": value}
```

### 2. Error Handling

Use structured logging with the logging utilities:

```python
from src.utils.logging_setup import get_current_logger, log_event

logger = get_current_logger()

try:
    # Agent logic
    log_event(logger, level="INFO", event="agent.step.success",
              phase="builder", node="my_agent",
              message="Step completed", payload={"key": "value"})
except Exception as exc:
    state.dag_state.error_log.append(f"My agent error: {exc}")
    log_event(logger, level="ERROR", event="agent.step.fail",
              phase="builder", node="my_agent",
              message=str(exc))
```

### 3. LLM Integration

For agents using LLM:

```python
from src.model import get_llm
from src.configuration import Configuration
from langchain_core.prompts import ChatPromptTemplate

llm = get_llm(Configuration.from_runnable_config(None))
prompt = ChatPromptTemplate.from_template(MY_PROMPT_TEMPLATE)
chain = prompt | llm

response = chain.invoke({"var1": value1})
```

### 4. Immutable State Updates

Always create new state objects rather than modifying in place:

```python
# Good: Immutable update
new_dag = state.dag_state.model_copy(update={"nodes": new_nodes})
return {"dag_state": new_dag}

# Avoid: Direct mutation (except for error logs)
state.dag_state.nodes[key] = value  # Side effect
```

### 5. Adding to Workflow

To add a new agent to the workflow, update [phm_outer_graph.py](../phm_outer_graph.py):

```python
# Add the node
builder.add_node("my_agent", lambda state: _run_node("my_agent", my_agent, state))

# Add edges
builder.add_edge("previous_node", "my_agent")
builder.add_edge("my_agent", "next_node")
```

---

## Related Files

- [../phm_outer_graph.py](../phm_outer_graph.py) - Graph orchestration and workflow definitions
- [../states/phm_states.py](../states/phm_states.py) - State management (PHMState, DAGState)
- [../tools/](../tools/) - Signal processing operators used by Execute Agent
- [../prompts/](../prompts/) - LLM prompt templates for Plan, Reflect, Report agents
