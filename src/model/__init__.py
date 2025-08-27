"""
Multi-provider LLM integration for PHMGA system.
"""

from .providers import (
    get_provider_registry,
    create_llm_provider,
    get_llm_factory,
    LLMProviderFactory,
    LLMProviderConfig,
    BaseLLMProvider
)

# Import functions from root model.py file
try:
    from model import (
        get_llm,
        get_default_llm,
        get_llm_by_provider,
        list_available_providers,
        auto_select_llm
    )
    
    __all__ = [
        # Provider system
        'get_provider_registry',
        'create_llm_provider', 
        'get_llm_factory',
        'LLMProviderFactory',
        'LLMProviderConfig',
        'BaseLLMProvider',
        # Main LLM functions
        'get_llm',
        'get_default_llm',
        'get_llm_by_provider',
        'list_available_providers',
        'auto_select_llm'
    ]
    
except ImportError:
    # Fallback if model.py functions are not available
    __all__ = [
        'get_provider_registry',
        'create_llm_provider', 
        'get_llm_factory',
        'LLMProviderFactory',
        'LLMProviderConfig',
        'BaseLLMProvider'
    ]