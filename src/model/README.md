# PHMGA Model Integration

The model module provides LLM (Large Language Model) integration for the PHMGA system. It implements factory functions and configuration management for seamless integration of language models into the signal processing and analysis pipeline.

## Architecture Overview

The model system provides:
- **Factory Functions**: Centralized LLM instantiation with configuration
- **Provider Abstraction**: Support for multiple LLM providers (currently Google Gemini)
- **Configuration Management**: Automatic parameter resolution from environment and config
- **Testing Support**: Mock LLM for testing and development

## Core Components

### 1. LLM Factory Functions

#### Primary Factory (`get_llm`)

```python
def get_llm(
    configurable: Optional[Configuration] = None,
    *,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> ChatGoogleGenerativeAI:
    """
    Return a chat model instance for agent use.

    Parameters
    ----------
    configurable : Optional[Configuration]
        Configuration object providing model settings. If None, a new
        Configuration will be created with environment variables.
    temperature : float, optional
        Sampling temperature for the model. Defaults to 1.0.
    max_retries : int, optional
        Maximum number of API retries. Defaults to 2.

    Returns
    -------
    ChatGoogleGenerativeAI
        Instantiated LLM ready for calls.
    """
```

#### Default Factory (`get_default_llm`)

```python
def get_default_llm(
    config: Optional[Configuration] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Return a Gemini chat model for agent use.

    Parameters
    ----------
    config : Configuration, optional
        Optional configuration from which to read default model name.
    model_name : str, optional
        Explicit model name to instantiate. If omitted, config or
        environment variables provide query_generator_model.
    **kwargs : Any
        Additional parameters forwarded to ChatGoogleGenerativeAI.

    Returns
    -------
    ChatGoogleGenerativeAI
        Configured chat model instance ready for invocation.
    """
```

### 2. Configuration Integration

The model system integrates with the configuration module for centralized settings:

```python
from src.configuration import Configuration
from src.model import get_llm

# Automatic configuration from environment
llm = get_llm()

# Explicit configuration
config = Configuration(
    query_generator_model="gemini-2.5-pro",
    reflection_model="gemini-2.5-pro",
    answer_model="gemini-2.5-pro"
)
llm = get_llm(config)
```

### 3. Environment Variables

The system automatically reads configuration from environment variables:

```bash
# Required for Google Gemini
export GEMINI_API_KEY="your_api_key_here"

# Optional model overrides
export QUERY_GENERATOR_MODEL="gemini-2.5-pro"
export REFLECTION_MODEL="gemini-2.5-pro"
export ANSWER_MODEL="gemini-2.5-pro"
```

## Usage Examples

### Basic LLM Usage

```python
from src.model import get_llm
from langchain_core.prompts import ChatPromptTemplate

# Get configured LLM
llm = get_llm(temperature=0.7, max_retries=3)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Analyze the following signal processing task: {task}"
)

# Create chain
chain = prompt | llm

# Invoke with data
response = chain.invoke({"task": "bearing fault diagnosis"})
print(response.content)
```

### Structured Output

```python
from pydantic import BaseModel, Field
from typing import List

class AnalysisPlan(BaseModel):
    """Structured analysis plan."""
    steps: List[str] = Field(..., description="Processing steps")
    priority: str = Field(..., description="Priority level")

# Use structured output
llm = get_llm()
structured_llm = llm.with_structured_output(AnalysisPlan)

response = structured_llm.invoke("Create a plan for bearing analysis")
print(f"Steps: {response.steps}")
print(f"Priority: {response.priority}")
```

### Agent Integration

```python
from src.model import get_llm
from src.configuration import Configuration

def example_agent(state, config=None):
    """Example agent using LLM integration."""
    
    # Get configured LLM
    llm = get_llm(Configuration.from_runnable_config(config))
    
    # Use in agent logic
    prompt = f"Analyze this signal processing state: {state.user_instruction}"
    response = llm.invoke(prompt)
    
    return {"analysis": response.content}
```

## Testing Support

### Mock LLM for Testing

```python
from langchain_community.chat_models import FakeListChatModel
from src.model import _FAKE_LLM

# Set up mock responses for testing
test_responses = [
    "Mock analysis response",
    "Mock planning response",
    "Mock reflection response"
]

# Configure fake LLM
_FAKE_LLM = FakeListChatModel(responses=test_responses)

# Use in tests
def test_agent_with_mock_llm():
    # Agent will use mock responses
    result = agent_function(test_state)
    assert "Mock analysis" in result["analysis"]
```

### Test Configuration

```python
import os
from src.model import get_llm

# Test with environment override
os.environ["FAKE_LLM"] = "true"
llm = get_llm()  # Returns mock LLM

# Test with real LLM (requires API key)
os.environ.pop("FAKE_LLM", None)
os.environ["GEMINI_API_KEY"] = "test_key"
llm = get_llm()  # Returns real LLM
```

## Error Handling

### API Key Management

```python
import os
from src.model import get_llm

def safe_llm_creation():
    """Safely create LLM with error handling."""
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY not set, using mock LLM")
        os.environ["FAKE_LLM"] = "true"
    
    try:
        llm = get_llm()
        return llm
    except Exception as e:
        print(f"LLM creation failed: {e}")
        # Fallback to mock
        from langchain_community.chat_models import FakeListChatModel
        return FakeListChatModel(responses=["Fallback response"])
```

### Retry Logic

```python
from src.model import get_llm
import time

def robust_llm_call(prompt, max_attempts=3):
    """Make LLM call with retry logic."""
    
    llm = get_llm(max_retries=2)
    
    for attempt in range(max_attempts):
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Integration Patterns

### Configuration-Driven Usage

```python
from src.configuration import Configuration
from src.model import get_default_llm

# Different models for different purposes
config = Configuration()

# Planning model
planning_llm = get_default_llm(config, model_name=config.query_generator_model)

# Reflection model  
reflection_llm = get_default_llm(config, model_name=config.reflection_model)

# Answer model
answer_llm = get_default_llm(config, model_name=config.answer_model)
```

### Prompt Management

```python
from langchain_core.prompts import ChatPromptTemplate
from src.model import get_llm

class PromptManager:
    """Centralized prompt management."""
    
    def __init__(self):
        self.llm = get_llm()
        
    def create_analysis_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "Analyze the signal processing task:\n"
            "Task: {task}\n"
            "Context: {context}\n"
            "Provide structured analysis."
        )
        return prompt | self.llm
    
    def create_planning_chain(self):
        prompt = ChatPromptTemplate.from_template(
            "Create a processing plan for:\n"
            "Instruction: {instruction}\n"
            "Available tools: {tools}\n"
            "Return JSON plan."
        )
        return prompt | self.llm.with_structured_output(PlanSchema)
```

## Dependencies

### Required Packages

```python
# Core LangChain
langchain-core>=0.1.0
langchain-google-genai>=1.0.0

# Testing
langchain-community>=0.0.1  # For FakeListChatModel

# Configuration
pydantic>=2.0.0
```

### Environment Setup

```bash
# Install dependencies
pip install langchain-core langchain-google-genai langchain-community

# Set up API key
export GEMINI_API_KEY="your_gemini_api_key"

# Optional: Configure models
export QUERY_GENERATOR_MODEL="gemini-2.5-pro"
export REFLECTION_MODEL="gemini-2.5-pro"
export ANSWER_MODEL="gemini-2.5-pro"
```

## Future Extensions

The model system is designed for easy extension to support additional LLM providers:

```python
# Future multi-provider support
from src.model import get_llm

# Provider selection via configuration
config = Configuration(provider="openai", model="gpt-4")
llm = get_llm(config)

# Or via environment
os.environ["LLM_PROVIDER"] = "anthropic"
os.environ["LLM_MODEL"] = "claude-3"
llm = get_llm()
```

This model integration system provides a robust foundation for LLM usage throughout the PHMGA system while maintaining flexibility and testability.
