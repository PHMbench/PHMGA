# Multi-Provider LLM Integration

The PHMGA system now supports multiple LLM providers through a unified interface, enabling seamless switching between Google Gemini, OpenAI, and other providers while maintaining full backward compatibility.

## Overview

The multi-provider system consists of:
- **Abstract Base Class**: `BaseLLMProvider` defining the unified interface
- **Concrete Providers**: Google, OpenAI, and Mock implementations
- **Provider Registry**: Dynamic provider discovery and instantiation
- **Factory System**: Unified state integration and auto-detection
- **Backward Compatibility**: Seamless migration from legacy code

## Supported Providers

### 1. Google Gemini Provider

**Models Supported:**
- `gemini-2.5-pro` (recommended)
- `gemini-2.5-flash`
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-pro`
- `gemini-pro-vision`

**Configuration:**
```python
from src.model.providers import create_llm_provider

# Create Google provider
provider = create_llm_provider(
    provider="google",
    model="gemini-2.5-pro",
    api_key="your_gemini_api_key",  # or set GEMINI_API_KEY env var
    temperature=0.7,
    max_retries=3
)

# Use the provider
response = provider.invoke("Analyze this signal processing task")
print(response)
```

**Environment Variables:**
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### 2. OpenAI Provider

**Models Supported:**
- `gpt-4o` (recommended)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4`
- `gpt-3.5-turbo`
- `gpt-3.5-turbo-16k`

**Configuration:**
```python
# Create OpenAI provider
provider = create_llm_provider(
    provider="openai",
    model="gpt-4o",
    api_key="your_openai_api_key",  # or set OPENAI_API_KEY env var
    temperature=0.5,
    max_retries=2
)

# Use the provider
response = provider.invoke("Create a signal processing plan")
print(response)
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Mock Provider (Testing)

**Models Supported:**
- `mock-model`
- `test-model`

**Configuration:**
```python
# Create mock provider for testing
provider = create_llm_provider(
    provider="mock",
    model="mock-model",
    extra_params={
        "responses": [
            "Mock response 1",
            "Mock response 2",
            "Mock response 3"
        ]
    }
)

# Use the provider
response = provider.invoke("Test prompt")
print(response)  # Returns "Mock response 1"
```

## Usage Examples

### Basic Provider Usage

```python
from src.model.providers import create_llm_provider

# Method 1: Direct provider creation
google_provider = create_llm_provider("google", "gemini-2.5-pro")
openai_provider = create_llm_provider("openai", "gpt-4o")

# Method 2: Using the unified model interface
from src.model import get_llm_by_provider

google_llm = get_llm_by_provider("google", "gemini-2.5-pro", temperature=0.7)
openai_llm = get_llm_by_provider("openai", "gpt-4o", temperature=0.5)

# Method 3: Auto-detection based on available API keys
from src.model import auto_select_llm

best_llm = auto_select_llm(temperature=0.6)
```

### Unified State Integration

```python
from src.states.phm_states import get_unified_state
from src.model import get_llm

# Configure provider through unified state
unified_state = get_unified_state()
unified_state.set('llm.provider', 'openai')
unified_state.set('llm.model', 'gpt-4o')
unified_state.set('llm.openai_api_key', 'your_key')
unified_state.set('llm.temperature', 0.7)

# Get LLM using unified configuration
llm = get_llm()  # Automatically uses OpenAI with configured settings
```

### Structured Output

```python
from pydantic import BaseModel, Field
from src.model.providers import create_llm_provider

class AnalysisPlan(BaseModel):
    steps: list[str] = Field(..., description="Processing steps")
    priority: str = Field(..., description="Priority level")

# Create provider and configure for structured output
provider = create_llm_provider("google", "gemini-2.5-pro")
structured_llm = provider.with_structured_output(AnalysisPlan)

# Get structured response
plan = structured_llm.invoke("Create a bearing analysis plan")
print(f"Steps: {plan.steps}")
print(f"Priority: {plan.priority}")
```

### Provider Fallback

```python
from src.model.providers import get_llm_factory

factory = get_llm_factory()

# Create provider with fallback
provider = factory.create_with_fallback(
    preferred_provider="openai",
    model="gpt-4o",
    fallback_provider="google",
    temperature=0.7
)

# If OpenAI fails, automatically falls back to Google
response = provider.invoke("Analyze signal data")
```

### Agent Integration

```python
from src.model import get_llm
from src.states.phm_states import PHMState

def enhanced_plan_agent(state: PHMState) -> dict:
    """Plan agent with multi-provider support."""
    
    # Get LLM using unified state (automatically selects provider)
    llm = get_llm(temperature=0.8)
    
    # Or explicitly choose provider
    # llm = get_llm_by_provider("openai", "gpt-4o", temperature=0.8)
    
    # Use in agent logic
    prompt = f"Create processing plan for: {state.user_instruction}"
    response = llm.invoke(prompt)
    
    return {"plan": response}
```

## Configuration Management

### Environment Variables

The system automatically detects and uses environment variables:

```bash
# Provider selection
export LLM_PROVIDER="openai"  # or "google"
export LLM_MODEL="gpt-4o"     # or "gemini-2.5-pro"

# API keys
export GEMINI_API_KEY="your_gemini_key"
export OPENAI_API_KEY="your_openai_key"

# Optional settings
export LLM_TEMPERATURE="0.7"
export LLM_MAX_RETRIES="3"
export LLM_TIMEOUT="30"
```

### YAML Configuration

```yaml
# config/llm_config.yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7
  max_retries: 3
  timeout: 30
  openai_api_key: "${OPENAI_API_KEY}"
  
  # Provider-specific settings
  extra_params:
    top_p: 0.9
    frequency_penalty: 0.1
```

### Programmatic Configuration

```python
from src.states.phm_states import get_unified_state

unified_state = get_unified_state()

# Configure Google provider
unified_state.set('llm.provider', 'google')
unified_state.set('llm.model', 'gemini-2.5-pro')
unified_state.set('llm.gemini_api_key', 'your_key')

# Configure OpenAI provider
unified_state.set('llm.provider', 'openai')
unified_state.set('llm.model', 'gpt-4o')
unified_state.set('llm.openai_api_key', 'your_key')

# Advanced settings
unified_state.set('llm.temperature', 0.8)
unified_state.set('llm.max_retries', 5)
unified_state.set('llm.extra_params', {'top_p': 0.9})
```

## Migration Guide

### From Legacy Google-Only System

**Old Code:**
```python
from src.configuration import Configuration
from src.model import get_default_llm

config = Configuration(phm_model="gemini-2.5-pro")
llm = get_default_llm(config)
```

**New Code:**
```python
from src.model import get_llm
from src.states.phm_states import get_unified_state

# Option 1: Use unified state
unified_state = get_unified_state()
unified_state.set('llm.provider', 'google')
unified_state.set('llm.model', 'gemini-2.5-pro')
llm = get_llm()

# Option 2: Direct provider access
from src.model import get_llm_by_provider
llm = get_llm_by_provider("google", "gemini-2.5-pro")
```

### Backward Compatibility

The system maintains full backward compatibility:

```python
# These still work (with deprecation warnings)
from src.configuration import Configuration
from src.model import get_default_llm

config = Configuration()
llm = get_default_llm(config)  # Still works, uses unified state internally
```

## Error Handling

### Provider Validation

```python
from src.model.providers import create_llm_provider

try:
    provider = create_llm_provider("google", "invalid-model")
except ValueError as e:
    print(f"Invalid configuration: {e}")

# Check validation before creation
from src.model.providers import LLMProviderConfig, GoogleLLMProvider

config = LLMProviderConfig(
    provider="google",
    model="gemini-2.5-pro",
    api_key="test_key"
)

provider = GoogleLLMProvider(config)
errors = provider.validate_config()
if errors:
    print(f"Configuration errors: {errors}")
```

### Graceful Fallbacks

```python
from src.model import auto_select_llm
import warnings

# Automatically handles missing API keys
try:
    llm = auto_select_llm()
except Exception as e:
    print(f"LLM creation failed: {e}")
    # System automatically falls back to mock provider
```

## Best Practices

1. **Use Unified State**: Leverage the unified state management for consistent configuration
2. **Environment Variables**: Store API keys in environment variables, not code
3. **Provider Fallbacks**: Always configure fallback providers for production systems
4. **Structured Output**: Use Pydantic schemas for reliable LLM responses
5. **Error Handling**: Implement proper error handling and validation
6. **Testing**: Use the mock provider for unit tests and development

## Troubleshooting

### Common Issues

**Issue**: "Provider 'openai' not found"
**Solution**: Ensure `langchain-openai` is installed: `pip install langchain-openai`

**Issue**: "API key required for google"
**Solution**: Set the `GEMINI_API_KEY` environment variable

**Issue**: "Model gpt-5 not supported by openai"
**Solution**: Use a supported model name (see provider documentation above)

### Debug Mode

```python
from src.model.providers import get_provider_registry

# List all available providers and models
registry = get_provider_registry()
for provider_name in registry.list_providers():
    provider_class = registry.get_provider(provider_name)
    print(f"{provider_name}: {provider_class}")
```

This multi-provider system provides a robust, flexible foundation for LLM integration throughout the PHMGA system while maintaining full backward compatibility.
