# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PHMGA (Prognostics and Health Management Graph Agent) is an advanced framework for bearing fault diagnosis that combines LangGraph-based multi-agent systems with autonomous signal processing capabilities. The system uses a dual-layer architecture with LangGraph for workflow orchestration and dynamic computational graphs for signal analysis.

## Quick Start

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Environment variables (.env file)
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
LLM_PROVIDER=google  # or openai
LLM_MODEL=gemini-2.5-pro  # or gpt-4o
```

### Running Analysis Cases
```bash
# Run specific cases
python main.py case1 --config config/case1.yaml
python -m src.cases.case1
python -m src.cases.case2_predictive

# Run tests
python -m pytest tests/
python test_complete_system.py
```

## Architecture Overview

**Dual-Layer Architecture:**
- **Outer Layer**: LangGraph workflow management (START → Plan → Execute → Reflect → Report → END)
- **Inner Layer**: Dynamic signal processing DAG (Input → Operators → Features → Classification)

**Core Components:**
- Multi-agent system with 7+ specialized agents
- 60+ signal processing operators with automatic registration
- Comprehensive state management with immutable updates
- Multi-provider LLM integration (Google/OpenAI)
- Advanced visualization and reporting capabilities

## Documentation Index

### Core System Components

| Component | Location | Description |
|-----------|----------|-------------|
| **Agent System** | [`src/agents/CLAUDE.md`](src/agents/CLAUDE.md) | Multi-agent architecture, API references, Research agents |
| **Signal Processing** | [`src/tools/CLAUDE.md`](src/tools/CLAUDE.md) | 60+ operators, categories, custom development |
| **State Management** | [`src/states/CLAUDE.md`](src/states/CLAUDE.md) | PHMState, DAGState, node types, persistence |
| **LLM Integration** | [`src/model/CLAUDE.md`](src/model/CLAUDE.md) | Multi-provider support, configuration, structured output |
| **Prompt Templates** | [`src/prompts/CLAUDE.md`](src/prompts/CLAUDE.md) | PLANNER, EXECUTE, REFLECT, REPORT prompts |
| **Data Schemas** | [`src/schemas/CLAUDE.md`](src/schemas/CLAUDE.md) | AnalysisInsight, Plan schemas, validation |
| **Utilities** | [`src/utils/CLAUDE.md`](src/utils/CLAUDE.md) | Data loading, preprocessing, visualization, persistence |
| **Workflow Graphs** | [`src/graph/CLAUDE.md`](src/graph/CLAUDE.md) | LangGraph workflows, Builder/Executor graphs |

### Case Studies and Examples

| Case | Location | Description |
|------|----------|-------------|
| **Case 1** | `src/cases/case1.py` | Basic bearing fault diagnosis |
| **Case 2** | `src/cases/case2_predictive.py` | Predictive maintenance analysis |
| **Ottawa Case** | `src/cases/case_ottawa.ipynb` | Jupyter notebook example |

### Configuration Files

| File | Purpose |
|------|---------|
| `config/*.yaml` | Case configuration files |
| `src/langgraph.json` | LangGraph workflow configuration |
| `requirements.txt` | Python dependencies |

## Key Commands Reference

```bash
# Development
python main.py {case_name} --config {config_file}
python -m src.cases.{case_name}

# Testing
python -m pytest tests/
python -m pytest tests/test_{agent_name}.py
python -m src.agents.{agent_name}  # Direct agent testing

# Utilities
python -c "from src.utils import initialize_state; print('Utils loaded')"
python -c "from src.tools.signal_processing_schemas import OP_REGISTRY; print(list(OP_REGISTRY.keys()))"
```

## Important Files and Locations

### Core System Files
- `main.py`: Entry point for case execution
- `src/phm_outer_graph.py`: Main workflow graph construction
- `src/states/phm_states.py`: Central state management (PHMState class: lines 154-572)
- `src/tools/signal_processing_schemas.py`: Operator registry and base classes

### Agent Implementation Files
- `src/agents/plan_agent.py`: LLM planning agent (main logic: lines 42-147)
- `src/agents/execute_agent.py`: Dynamic execution engine
- `src/agents/reflect_agent.py`: Quality assessment system
- `src/agents/report_agent.py`: Report generation system

### Configuration and Testing
- `config/`: YAML configuration files for different cases
- `tests/`: Comprehensive test suite with unit and integration tests
- `src/cases/`: Working examples and use cases

## Signal Dimension Conventions

- `B`: Batch size (批处理大小，即样本数量)
- `L`: Signal length (信号长度)
- `C`: Channels (通道数)
- `F`: Frequency dimension (频率轴维度)
- `T`: Time frames (时间轴维度)
- `S`: Wavelet scales (小波变换尺度轴维度)

## Development Guidelines

### Quick Development Patterns
```python
# Get configured LLM
from src.model import get_llm
llm = get_llm(temperature=0.7)

# Initialize system state
from src.utils import initialize_state
state = initialize_state(user_instruction="...", metadata_path="...", ...)

# Use signal processing operators
from src.tools.signal_processing_schemas import get_operator
op_class = get_operator("fft")
operator = op_class()
```

### Performance Tips
- Use `use_parallel=True` for multi-core processing
- Enable `use_cache=True` for repeated operations
- Use `quick_mode=True` for rapid prototyping
- Monitor memory with large signal datasets

## For Detailed Documentation

Each component has comprehensive documentation in its respective `CLAUDE.md` file in the `src/` subdirectories. Refer to the Documentation Index above for specific topics.

This modular documentation structure allows for better maintainability and focused development on specific components of the PHMGA system.