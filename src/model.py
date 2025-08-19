"""Factory for language model instances used by agents."""

from __future__ import annotations

import os
import warnings
from typing import Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import FakeListChatModel

from .configuration import Configuration


# Global fake LLM instance for testing
_FAKE_LLM = None


def get_llm(
    configurable: Optional[Configuration] = None,
    *,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> Union[BaseChatModel, ChatGoogleGenerativeAI]:
    """
    Return a chat model instance for agent use.

    This function integrates with the unified state management system
    and multi-provider LLM support while maintaining backward compatibility.

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

    try:
        # Try to use new multi-provider system with unified state
        from src.model.providers import get_llm_factory
        from src.states.phm_states import get_unified_state

        factory = get_llm_factory()
        unified_state = get_unified_state()

        # Override temperature and max_retries in unified state if provided
        if temperature != 1.0:
            unified_state.set('llm.temperature', temperature)
        if max_retries != 2:
            unified_state.set('llm.max_retries', max_retries)

        # Override with explicit configuration if provided
        if configurable:
            warnings.warn(
                "Passing Configuration object is deprecated. "
                "Use unified state management instead.",
                DeprecationWarning,
                stacklevel=2
            )
            # Update unified state with legacy config values
            unified_state.set('llm.model', configurable.phm_model)

        # Create provider from unified state
        provider = factory.create_from_unified_state(unified_state)
        return provider.client

    except ImportError:
        # Fallback to legacy Google-only implementation
        warnings.warn(
            "Multi-provider system not available. Using legacy Google-only implementation.",
            RuntimeWarning,
            stacklevel=2
        )

        # Legacy implementation
        try:
            from src.states.phm_states import get_unified_state
            unified_state = get_unified_state()

            # Get LLM configuration from unified state
            llm_config = unified_state.get_llm_config()
            model_name = llm_config.get('model', 'gemini-2.5-pro')
            api_key = llm_config.get('gemini_api_key') or os.getenv("GEMINI_API_KEY")

            # Override with explicit configuration if provided
            if configurable:
                model_name = configurable.phm_model

            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                max_retries=max_retries,
                api_key=api_key,
            )

        except ImportError:
            # Final fallback to legacy configuration
            conf = configurable or Configuration.from_runnable_config(None)
            return ChatGoogleGenerativeAI(
                model=conf.phm_model,
                temperature=temperature,
                max_retries=max_retries,
                api_key=os.getenv("GEMINI_API_KEY"),
            )


def get_default_llm(
    config: Optional[Configuration] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> ChatGoogleGenerativeAI:
    """
    Return a Gemini chat model for agent use.

    This function is deprecated. Use get_llm() instead.

    Parameters
    ----------
    config : Configuration, optional
        Optional configuration from which to read default model name.
    model_name : str, optional
        Explicit model name to instantiate. If omitted, ``config`` or
        environment variables provide ``query_generator_model``.
    **kwargs : Any
        Additional parameters forwarded to ``ChatGoogleGenerativeAI``.

    Returns
    -------
    ChatGoogleGenerativeAI
        Configured chat model instance ready for invocation.
    """
    warnings.warn(
        "get_default_llm is deprecated. Use get_llm() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    try:
        # Try to use unified state management
        from src.states.phm_states import get_unified_state
        unified_state = get_unified_state()

        # Get model name from unified state or explicit parameter
        if model_name:
            final_model_name = model_name
        else:
            llm_config = unified_state.get_llm_config()
            final_model_name = llm_config.get('query_generator_model', 'gemini-2.5-pro')

        # Override with config if provided
        if config:
            final_model_name = model_name or config.query_generator_model

        api_key = unified_state.get('llm.gemini_api_key') or os.getenv("GEMINI_API_KEY")

        return ChatGoogleGenerativeAI(
            model=final_model_name,
            temperature=kwargs.get('temperature', 1.0),
            max_retries=kwargs.get('max_retries', 2),
            api_key=api_key,
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_retries']}
        )

    except ImportError:
        # Fallback to legacy behavior
        conf = config or Configuration.from_runnable_config(None)
        name = model_name or conf.query_generator_model
        return ChatGoogleGenerativeAI(
            model=name,
            temperature=kwargs.get('temperature', 1.0),
            max_retries=kwargs.get('max_retries', 2),
            api_key=os.getenv("GEMINI_API_KEY"),
            **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_retries']}
        )


def get_llm_by_provider(
    provider: str,
    model: str,
    **kwargs
) -> BaseChatModel:
    """
    Get LLM instance by explicit provider and model.

    This function provides direct access to specific providers without
    going through the unified state management system.

    Parameters
    ----------
    provider : str
        Provider name (google, openai, mock)
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
    try:
        from src.model.providers import create_llm_provider
        provider_instance = create_llm_provider(provider, model, **kwargs)
        return provider_instance.client
    except ImportError:
        raise RuntimeError(
            "Multi-provider system not available. "
            "Use get_llm() for legacy Google-only support."
        )


def list_available_providers() -> dict:
    """
    List all available LLM providers and their supported models.

    Returns
    -------
    dict
        Dictionary mapping provider names to their supported models

    Examples
    --------
    >>> providers = list_available_providers()
    >>> print(providers)
    {
        'google': ['gemini-2.5-pro', 'gemini-2.5-flash', ...],
        'openai': ['gpt-4o', 'gpt-4o-mini', ...],
        'mock': ['mock-model', 'test-model']
    }
    """
    try:
        from src.model.providers import get_provider_registry
        registry = get_provider_registry()

        providers_info = {}
        for provider_name in registry.list_providers():
            provider_class = registry.get_provider(provider_name)
            # Create a dummy instance to get supported models
            dummy_config = type('Config', (), {
                'provider': provider_name,
                'model': 'dummy',
                'api_key': 'dummy'
            })()
            dummy_instance = provider_class(dummy_config)
            providers_info[provider_name] = dummy_instance.supported_models

        return providers_info
    except ImportError:
        return {"google": ["gemini-2.5-pro", "gemini-1.5-pro"]}  # Legacy fallback


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
    try:
        from src.model.providers import get_llm_factory
        factory = get_llm_factory()
        provider = factory.auto_detect_provider()
        return provider.client
    except ImportError:
        # Fallback to legacy get_llm
        return get_llm(**kwargs)
