# PHMGA LLM Integration Documentation

This document provides comprehensive guidance for working with the PHMGA multi-provider LLM system.

## Multi-Provider Support

The system supports multiple LLM providers with unified interface:

### Supported Providers

#### Google Gemini
- **Models**: gemini-2.5-pro, gemini-2.5-flash, gemini-1.5-pro, gemini-1.5-flash
- **Environment Variable**: `GEMINI_API_KEY`

#### OpenAI
- **Models**: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- **Environment Variable**: `OPENAI_API_KEY`

#### DashScope (Qwen Models)
- **Models**: qwen-plus, qwen-turbo, qwen-max, qwen-max-longcontext, qwen2.5-coder-32b-instruct
- **Environment Variable**: `DASHSCOPE_API_KEY`
- **Features**: OpenAI-compatible API, thinking process control, latest Qwen models

#### Mock (Testing)
- **Models**: mock-model, test-model
- **Purpose**: Testing and development

## Configuration

### Environment Variables

```bash
# Provider selection
export LLM_PROVIDER="google"  # or "openai", "dashscope"
export LLM_MODEL="gemini-2.5-pro"  # or "gpt-4o", "qwen-plus"

# API keys
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_API_KEY="your_openai_key"
export DASHSCOPE_API_KEY="your_dashscope_key"

# Optional settings
export LLM_TEMPERATURE="0.7"
export LLM_MAX_RETRIES="3"
```

### Provider Usage

```python
from src.model import get_llm, get_llm_by_provider

# Auto-detect provider based on available API keys
llm = get_llm(temperature=0.7)

# Explicitly choose provider  
google_llm = get_llm_by_provider("google", "gemini-2.5-pro")
openai_llm = get_llm_by_provider("openai", "gpt-4o")
dashscope_llm = get_llm_by_provider("dashscope", "qwen-plus")
```

### Structured Output

```python
from src.schemas.plan_schema import AnalysisPlan

llm = get_llm()
structured_llm = llm.with_structured_output(AnalysisPlan)
plan = structured_llm.invoke("Create analysis plan")
```

### DashScope-Specific Features

```python
from src.model.providers import create_llm_provider

# Basic DashScope usage
dashscope_provider = create_llm_provider("dashscope", "qwen-plus")
llm = dashscope_provider.client

# With thinking process control (for Qwen3 models)
dashscope_with_thinking = create_llm_provider(
    "dashscope", 
    "qwen-plus",
    extra_params={"enable_thinking": False}  # Disable thinking process
)

# Advanced configuration
advanced_dashscope = create_llm_provider(
    "dashscope",
    "qwen-max",
    temperature=0.8,
    max_retries=3,
    timeout=60,
    extra_params={
        "enable_thinking": True,
        "top_p": 0.9
    }
)
```

### Unified State Configuration

```python
from src.states.phm_states import get_unified_state

unified_state = get_unified_state()
unified_state.set('llm.provider', 'google')
unified_state.set('llm.model', 'gemini-2.5-pro')
unified_state.set('llm.temperature', 0.8)
```

## Agent Integration

```python
from src.model import get_llm

def enhanced_agent(state: PHMState) -> dict:
    """Agent with multi-provider LLM support."""
    
    # Get LLM using unified state
    llm = get_llm(temperature=0.8)
    
    # Use in agent logic
    response = llm.invoke(f"Process: {state.user_instruction}")
    
    return {"response": response}
```

## Error Handling

```python
from src.model import auto_select_llm

try:
    llm = auto_select_llm()
except Exception as e:
    print(f"LLM creation failed: {e}")
    # System automatically falls back to mock provider
```

This multi-provider system provides a robust, flexible foundation for LLM integration throughout the PHMGA system.