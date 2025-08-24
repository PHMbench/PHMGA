# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PHMGA (Prognostics and Health Management Graph Agent) is a LangGraph-based multi-agent system for bearing fault diagnosis and signal processing. The system uses a dual-layer architecture: an outer static workflow graph managed by LangGraph and an inner dynamic computational DAG for signal processing.

## Environment Setup and Dependencies

### Python Environment
- Python 3.12.9 (Anaconda recommended)
- Virtual environment recommended: `python -m venv venv_nvta`

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Environment Variables (.env file)
```bash
# Google/Gemini API (required - OpenAI not supported in NVTA branch)
GOOGLE_API_KEY="your_key"
GEMINI_API_KEY="your_key"  
USE_REAL_LLM="1"

# Optional - LangChain tracing
LANGCHAIN_API_KEY="your_key"
LANGCHAIN_TRACING_V2=true
```

## Common Development Commands

### Running Cases
```bash
# Run specific case with config
python main.py case1 --config config/case1.yaml

# Run case directly
python -m src.cases.case1

# Run from Jupyter notebooks
jupyter notebook src/cases/case1.ipynb
```

### Testing Agents Individually
```bash
# Test plan agent
python src/agents/plan_agent.py

# Test execute agent  
python src/agents/execute_agent.py
```

### Debugging LangGraph Workflows
```bash
# Enable tracing (set in code or env)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="phmga-debug"
```

## High-Level Architecture

### Dual-Layer System Design

**Outer Layer (Static Workflow):**
- Managed by LangGraph StateGraph
- Flow: `START → Plan → Execute → Reflect → (Plan|Report) → END`
- Defined in `src/phm_outer_graph.py`
- Two main graphs:
  - `build_builder_graph()`: Iterative DAG construction with plan-execute-reflect loop
  - `build_executor_graph()`: Final execution with inquire-prepare-train-report pipeline

**Inner Layer (Dynamic DAG):**
- Signal processing computational graph
- Nodes represent operations from `src/tools/`:
  - `expand_schemas.py`: Dimensionality expansion (STFT, Wavelet, etc.)
  - `transform_schemas.py`: Signal transformations  
  - `aggregate_schemas.py`: Feature extraction
  - `decision_schemas.py`: Classification operations
  - `multi_schemas.py`: Multi-variable operations

### Core Agent System

Located in `src/agents/`, each agent has specific responsibilities:

1. **plan_agent.py**: Decomposes user instructions into structured processing plans using LLM
2. **execute_agent.py**: Implements think-act loops to build DAG by applying operators
3. **reflect_agent.py**: Evaluates DAG quality and decides if revision needed
4. **report_agent.py**: Generates comprehensive analysis reports
5. **inquirer_agent.py**: Performs similarity analysis between signals
6. **dataset_preparer_agent.py**: Assembles features into ML-ready datasets
7. **shallow_ml_agent.py**: Trains and evaluates ML models

### State Management

Central state in `src/states/phm_states.py`:

- **PHMState**: Main workflow state containing user instruction, signals, DAG state, plans, and results
- **DAGState**: Topology-only representation of computational graph
- **Node Types**: InputData, ProcessedData, DataSetNode - represent different stages

State updates are immutable - agents return dictionaries with updated fields that get merged.

### Signal Processing Operators

All operators in `src/tools/` follow the PHMOperator base class pattern:
- Auto-registered in OP_REGISTRY
- Accessed via `get_operator(op_name)`
- Categories: Expand, Transform, Aggregate, Decision, Multi-variable
- Each implements `execute()` method for numpy array processing

## Research Cases (NVTA 2025 Version)

Current focus areas defined in README:
1. **Case 1**: DAG first - Basic signal processing pipeline
2. **Case 2**: DAG with data feature reflection - Adaptive processing
3. **Case 3**: Decouple DAG and data - Separate topology from computation
4. **Case 4**: Coding augmentation - LLM-assisted code generation
5. **Case 5**: Prompt evaluation - Optimize LLM prompting strategies
6. **Case 6**: Multi-agent concurrency - Parallel agent execution

## Project Structure

```
src/
├── agents/          # Core agent implementations
├── cases/           # Runnable case studies  
├── graph/           # Graph construction utilities
├── model/           # LLM configuration (model.py)
├── prompts/         # Agent prompt templates
├── schemas/         # Data schemas and plans
├── states/          # State definitions
├── tools/           # Signal processing operators
└── utils/           # Helper functions
```

## Key Technical Patterns

### Adding New Operators
1. Create class inheriting from appropriate base (ExpandOp, TransformOp, etc.)
2. Implement `execute()` method
3. Use `@register_op` decorator for auto-registration
4. Operator automatically available to agents

### LLM Provider Configuration
- **NVTA Branch**: Only Google/Gemini supported (OpenAI removed)
- Model selection: `LLM_MODEL=gemini-2.0-flash-exp`
- Configuration in `src/configuration.py` and `src/model/__init__.py`

### DAG Persistence
- States saved via pickle: `save_state(state, path)`
- Results saved to `save/` directory structure
- Visualizations saved as PNG files

## Current Development Focus

The **NVTA_2025_Version branch** is a minimal demo focusing on core PHM Graph Agent functionality. Changes from main branch:
- **Google API only** (OpenAI support removed)
- **Minimal dependencies** and simplified structure
- **Core demo cases** (case_exp2, case_exp2.5, case_exp_ottawa)
- **Removed research modules** and redundant files
- **Streamlined for demonstration purposes**

## Important Files

- `main.py`: Entry point for case execution
- `src/phm_outer_graph.py`: Core workflow definitions
- `src/agents/plan_agent.py`: LLM planning logic (lines 42-148)
- `src/agents/execute_agent.py`: DAG construction logic
- `src/states/phm_states.py`: State definitions (lines 88-200)
- `src/tools/signal_processing_schemas.py`: Operator registry and base classes