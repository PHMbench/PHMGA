"""Factory for language model instances used by agents."""

from __future__ import annotations

import os
import warnings
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models import FakeListChatModel

from ..configuration import Configuration

# Import research LLM factory components
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'tutorials_research', 'Part1_Foundations', 'modules'))
    from llm_providers import (
        LLMProvider, 
        ResearchLLMFactory, 
        create_research_llm,
        list_research_providers
    )
    RESEARCH_LLM_AVAILABLE = True
except ImportError:
    # Fallback imports for legacy support
    from langchain_google_genai import ChatGoogleGenerativeAI
    RESEARCH_LLM_AVAILABLE = False
    warnings.warn(
        "Research LLM providers not available. Using legacy Google-only implementation.",
        RuntimeWarning,
        stacklevel=2
    )


# Global fake LLM instance for testing
_FAKE_LLM = None


# Convenience functions for accessing research LLM capabilities
def print_available_providers():
    """Print available LLM providers and their status."""
    if RESEARCH_LLM_AVAILABLE:
        list_research_providers()
    else:
        providers = list_available_providers()
        print("🔍 Available LLM Providers:")
        print("-" * 40)
        for name, info in providers.items():
            status = "✅" if info['available'] else "❌"
            print(f"{status} {name.upper():<10} - {info['description']}")
            print(f"   Models: {', '.join(info['models'])}")
            print()


def get_llm(
    configurable: Optional[Configuration] = None,
    *,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> Union[BaseChatModel, 'ChatGoogleGenerativeAI']:
    """
    Return a chat model instance for agent use.

    Uses multi-provider LLM system with automatic provider detection
    and fallback support for testing and legacy configurations.

    Parameters
    ----------
    configurable : Optional[Configuration]
        Configuration object providing model settings. If None, auto-detects
        the best available provider.
    temperature : float, optional
        Sampling temperature for the model. Defaults to 1.0.
    max_retries : int, optional
        Maximum number of API retries. Defaults to 2.

    Returns
    -------
    Union[BaseChatModel, ChatGoogleGenerativeAI]
        Instantiated LLM ready for calls.
    """
    global _FAKE_LLM

    # Check for fake LLM mode (testing)
    if os.getenv("FAKE_LLM", "").lower() in ("true", "1", "yes"):
        if _FAKE_LLM is None:
            _FAKE_LLM = FakeListChatModel(responses=[
                "This is a mock response for testing.",
                "Another mock response.",
                "Final mock response."
            ])
        return _FAKE_LLM

    if RESEARCH_LLM_AVAILABLE:
        # Use research LLM factory for multi-provider support
        try:
            # Determine provider and model from configuration
            provider_name = None
            model_name = None
            
            if configurable:
                # Extract provider info from legacy configuration
                model_name = configurable.phm_model
                if "gemini" in model_name.lower():
                    provider_name = "google"
                elif "gpt" in model_name.lower():
                    provider_name = "openai"
                elif "qwen" in model_name.lower():
                    provider_name = "dashscope"
                elif "glm" in model_name.lower():
                    provider_name = "zhipuai"
            
            # Create LLM using research factory
            return create_research_llm(
                provider_name=provider_name,
                model=model_name,
                temperature=temperature,
                **({'max_tokens': max_retries} if provider_name == "google" else {'max_retries': max_retries})
            )
            
        except Exception as e:
            warnings.warn(
                f"Failed to create research LLM: {e}. Falling back to legacy implementation.",
                RuntimeWarning,
                stacklevel=2
            )
    
    # Fallback to legacy Google-only implementation
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    # Determine model name
    if configurable:
        model_name = configurable.phm_model
    else:
        model_name = os.getenv("LLM_MODEL", "gemini-2.5-pro")
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please set it in your environment or .env file."
        )
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
    )


def get_default_llm(
    config: Optional[Configuration] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> Union[BaseChatModel, 'ChatGoogleGenerativeAI']:
    """
    Return a chat model for agent use.

    This function is deprecated. Use get_llm() instead.

    Parameters
    ----------
    config : Configuration, optional
        Optional configuration from which to read default model name.
    model_name : str, optional
        Explicit model name to instantiate. If omitted, uses default model.
    **kwargs : Any
        Additional parameters forwarded to the LLM.

    Returns
    -------
    Union[BaseChatModel, ChatGoogleGenerativeAI]
        Configured chat model instance ready for invocation.
    """
    warnings.warn(
        "get_default_llm is deprecated. Use get_llm() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Create a temporary configuration if needed
    if model_name or config:
        temp_config = config or Configuration.from_runnable_config(None)
        if model_name:
            # Create new config with specified model
            temp_config = Configuration(
                phm_model=model_name,
                query_generator_model=model_name
            )
        return get_llm(configurable=temp_config, **kwargs)
    else:
        return get_llm(**kwargs)


def get_llm_by_provider(
    provider: str,
    model: str,
    **kwargs
) -> BaseChatModel:
    """
    Get LLM instance by explicit provider and model.

    Parameters
    ----------
    provider : str
        Provider name (google, openai, dashscope, zhipuai, mock)
    model : str
        Model name
    **kwargs : Any
        Additional configuration parameters

    Returns
    -------
    BaseChatModel
        LLM instance for the specified provider

    Examples
    --------
    >>> # Get Google Gemini model
    >>> llm = get_llm_by_provider("google", "gemini-2.5-pro")

    >>> # Get OpenAI model
    >>> llm = get_llm_by_provider("openai", "gpt-4o", temperature=0.7)

    >>> # Get mock model for testing
    >>> llm = get_llm_by_provider("mock", "mock-model")
    """
    if provider == "mock":
        return FakeListChatModel(responses=[
            "Mock response for testing.",
            "Another mock response."
        ])
    
    if RESEARCH_LLM_AVAILABLE:
        try:
            return create_research_llm(
                provider_name=provider,
                model=model,
                **kwargs
            )
        except Exception as e:
            warnings.warn(
                f"Failed to create {provider} LLM: {e}",
                RuntimeWarning,
                stacklevel=2
            )
    
    # Fallback for Google provider only
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found for Google provider")
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            **kwargs
        )
    
    raise RuntimeError(
        f"Provider '{provider}' not available. "
        "Install research LLM providers or use 'google' provider."
    )


def list_available_providers() -> dict:
    """
    List all available LLM providers and their supported models.

    Returns
    -------
    dict
        Dictionary mapping provider names to their availability and models

    Examples
    --------
    >>> providers = list_available_providers()
    >>> print(providers)
    {
        'google': {
            'available': True,
            'models': ['gemini-2.5-pro', 'gemini-2.5-flash']
        },
        'openai': {
            'available': False,
            'models': ['gpt-4o', 'gpt-4o-mini']
        }
    }
    """
    if RESEARCH_LLM_AVAILABLE:
        try:
            factory = ResearchLLMFactory()
            available_providers = factory.get_available_providers()
            
            providers_info = {}
            for provider, config in available_providers.items():
                providers_info[provider.value] = {
                    'available': config['available'],
                    'models': [config['default_model'], config['fast_model']],
                    'description': config['description']
                }
            
            # Add mock provider
            providers_info['mock'] = {
                'available': True,
                'models': ['mock-model', 'test-model'],
                'description': 'Mock provider for testing'
            }
            
            return providers_info
        except Exception:
            pass
    
    # Legacy fallback
    google_available = bool(os.getenv("GEMINI_API_KEY"))
    return {
        "google": {
            'available': google_available,
            'models': ["gemini-2.5-pro", "gemini-1.5-pro"],
            'description': 'Google Gemini models'
        },
        "mock": {
            'available': True,
            'models': ['mock-model', 'test-model'],
            'description': 'Mock provider for testing'
        }
    }


def auto_select_llm(**kwargs) -> BaseChatModel:
    """
    Automatically select the best available LLM provider.

    This function checks for available API keys and selects the best
    provider automatically, with fallback to mock for testing.

    Parameters
    ----------
    **kwargs : Any
        Additional configuration parameters

    Returns
    -------
    BaseChatModel
        Best available LLM instance

    Examples
    --------
    >>> # Automatically select best provider
    >>> llm = auto_select_llm(temperature=0.7)
    """
    if RESEARCH_LLM_AVAILABLE:
        try:
            factory = ResearchLLMFactory()
            recommended_provider = factory.get_recommended_provider()
            
            if recommended_provider:
                return create_research_llm(
                    provider_name=recommended_provider.value,
                    **kwargs
                )
        except Exception as e:
            warnings.warn(
                f"Failed to auto-select research LLM: {e}",
                RuntimeWarning,
                stacklevel=2
            )
    
    # Fallback to standard get_llm
    return get_llm(**kwargs)
