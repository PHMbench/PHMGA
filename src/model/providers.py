"""
Multi-provider LLM integration for PHMGA system.

This module provides a unified interface for multiple LLM providers,
enabling seamless switching between Google Gemini, OpenAI, and other providers
while maintaining backward compatibility.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Union
import os
import warnings
from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    
    provider: str = Field(..., description="Provider name (google, openai, etc.)")
    model: str = Field(..., description="Model name")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    temperature: float = Field(1.0, description="Sampling temperature")
    max_retries: int = Field(2, description="Maximum number of retries")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    This class defines the unified interface that all LLM providers must implement,
    ensuring consistent behavior across different providers.
    """
    
    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._client: Optional[BaseChatModel] = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of supported model names."""
        pass
    
    @abstractmethod
    def _create_client(self) -> BaseChatModel:
        """Create and return the LLM client instance."""
        pass
    
    @property
    def client(self) -> BaseChatModel:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client
    
    def validate_config(self) -> List[str]:
        """
        Validate provider configuration.
        
        Returns
        -------
        List[str]
            List of validation errors, empty if valid
        """
        errors = []
        
        if not self.config.api_key:
            errors.append(f"API key required for {self.provider_name}")
        
        if self.config.model not in self.supported_models:
            errors.append(f"Model {self.config.model} not supported by {self.provider_name}")
        
        if not 0 <= self.config.temperature <= 2:
            errors.append("Temperature must be between 0 and 2")
        
        return errors
    
    def invoke(self, messages: Union[str, List[BaseMessage]], **kwargs) -> str:
        """
        Invoke the LLM with messages.
        
        Parameters
        ----------
        messages : Union[str, List[BaseMessage]]
            Input messages or string
        **kwargs : Any
            Additional parameters
        
        Returns
        -------
        str
            LLM response content
        """
        try:
            if isinstance(messages, str):
                response = self.client.invoke(messages, **kwargs)
            else:
                response = self.client.invoke(messages, **kwargs)
            
            return response.content
            
        except Exception as e:
            raise RuntimeError(f"LLM invocation failed for {self.provider_name}: {e}") from e
    
    def with_structured_output(self, schema: Type[BaseModel]) -> BaseChatModel:
        """
        Get client configured for structured output.
        
        Parameters
        ----------
        schema : Type[BaseModel]
            Pydantic schema for structured output
        
        Returns
        -------
        BaseChatModel
            Client configured for structured output
        """
        return self.client.with_structured_output(schema)


class GoogleLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-pro",
            "gemini-pro-vision"
        ]
    
    def _create_client(self) -> BaseChatModel:
        """Create Google Gemini client."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Google Gemini provider requires langchain-google-genai. "
                "Install with: pip install langchain-google-genai"
            ) from e
        
        return ChatGoogleGenerativeAI(
            model=self.config.model,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "openai"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    def _create_client(self) -> BaseChatModel:
        """Create OpenAI client."""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI provider requires langchain-openai. "
                "Install with: pip install langchain-openai"
            ) from e
        
        return ChatOpenAI(
            model=self.config.model,
            openai_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class TongyiProvider(BaseLLMProvider):
    """Tongyi/Qwen LLM provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "tongyi"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "qwen-turbo",
            "qwen-plus", 
            "qwen-max",
            "qwen-max-longcontext"
        ]
    
    def _create_client(self) -> BaseChatModel:
        """Create Tongyi client."""
        try:
            from langchain_community.llms import Tongyi
        except ImportError as e:
            raise ImportError(
                "Tongyi provider requires langchain-community and dashscope. "
                "Install with: pip install langchain-community dashscope"
            ) from e
        
        return Tongyi(
            model=self.config.model,
            dashscope_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class GLMProvider(BaseLLMProvider):
    """GLM (智谱) LLM provider implementation."""
    
    @property
    def provider_name(self) -> str:
        return "glm"
    
    @property
    def supported_models(self) -> List[str]:
        return [
            "glm-4",
            "glm-4-air",
            "glm-4-flash", 
            "glm-3-turbo"
        ]
    
    def _create_client(self) -> BaseChatModel:
        """Create GLM client."""
        try:
            from langchain_community.chat_models import ChatZhipuAI
        except ImportError as e:
            raise ImportError(
                "GLM provider requires langchain-community and zhipuai. "
                "Install with: pip install langchain-community zhipuai"
            ) from e
        
        return ChatZhipuAI(
            model=self.config.model,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **self.config.extra_params
        )


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    @property
    def supported_models(self) -> List[str]:
        return ["mock-model", "test-model"]
    
    def _create_client(self) -> BaseChatModel:
        """Create mock client."""
        try:
            from langchain_community.chat_models import FakeListChatModel
        except ImportError as e:
            raise ImportError(
                "Mock provider requires langchain-community. "
                "Install with: pip install langchain-community"
            ) from e
        
        responses = self.config.extra_params.get('responses', [
            "This is a mock response for testing.",
            "Another mock response.",
            "Final mock response."
        ])
        
        return FakeListChatModel(responses=responses)


class LLMProviderRegistry:
    """Registry for LLM providers."""
    
    def __init__(self):
        self._providers: Dict[str, Type[BaseLLMProvider]] = {}
        self._register_default_providers()
    
    def _register_default_providers(self):
        """Register default providers."""
        self.register("google", GoogleLLMProvider)
        self.register("openai", OpenAIProvider)
        self.register("tongyi", TongyiProvider)
        self.register("glm", GLMProvider)
        self.register("mock", MockLLMProvider)
    
    def register(self, name: str, provider_class: Type[BaseLLMProvider]):
        """
        Register a new provider.
        
        Parameters
        ----------
        name : str
            Provider name
        provider_class : Type[BaseLLMProvider]
            Provider class
        """
        self._providers[name] = provider_class
    
    def get_provider(self, name: str) -> Type[BaseLLMProvider]:
        """
        Get provider class by name.
        
        Parameters
        ----------
        name : str
            Provider name
        
        Returns
        -------
        Type[BaseLLMProvider]
            Provider class
        
        Raises
        ------
        ValueError
            If provider not found
        """
        if name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available: {available}")
        
        return self._providers[name]
    
    def list_providers(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())
    
    def create_provider(self, config: LLMProviderConfig) -> BaseLLMProvider:
        """
        Create provider instance from configuration.
        
        Parameters
        ----------
        config : LLMProviderConfig
            Provider configuration
        
        Returns
        -------
        BaseLLMProvider
            Provider instance
        """
        provider_class = self.get_provider(config.provider)
        return provider_class(config)


# Global provider registry
_provider_registry = LLMProviderRegistry()


def get_provider_registry() -> LLMProviderRegistry:
    """Get the global provider registry."""
    return _provider_registry


def register_provider(name: str, provider_class: Type[BaseLLMProvider]):
    """Register a new provider globally."""
    _provider_registry.register(name, provider_class)


def create_llm_provider(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """
    Create LLM provider instance.

    Parameters
    ----------
    provider : str
        Provider name (google, openai, mock)
    model : str
        Model name
    api_key : str, optional
        API key (will try to get from environment if not provided)
    **kwargs : Any
        Additional configuration parameters

    Returns
    -------
    BaseLLMProvider
        Provider instance
    """
    # Auto-detect API key from environment if not provided
    if api_key is None:
        if provider == "google":
            api_key = os.getenv("GEMINI_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "tongyi":
            api_key = os.getenv("DASHSCOPE_API_KEY")
        elif provider == "glm":
            api_key = os.getenv("ZHIPUAI_API_KEY")

    config = LLMProviderConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        **kwargs
    )

    return _provider_registry.create_provider(config)


class LLMProviderFactory:
    """
    Factory for creating LLM providers with unified state integration.

    This factory integrates with the UnifiedStateManager to provide
    seamless configuration management and provider selection.
    """

    def __init__(self):
        self.registry = get_provider_registry()

    def create_from_unified_state(self, unified_state=None) -> BaseLLMProvider:
        """
        Create LLM provider from unified state configuration.

        Parameters
        ----------
        unified_state : UnifiedStateManager, optional
            Unified state manager instance. If None, will get global instance.

        Returns
        -------
        BaseLLMProvider
            Configured provider instance
        """
        if unified_state is None:
            try:
                from src.states.phm_states import get_unified_state
                unified_state = get_unified_state()
            except ImportError:
                raise RuntimeError("UnifiedStateManager not available")

        # Get LLM configuration from unified state
        llm_config = unified_state.get_llm_config()

        # Determine provider and model
        provider = llm_config.get('provider', 'google')
        model = llm_config.get('model', 'gemini-2.5-pro')

        # Get provider-specific API key
        if provider == 'google':
            api_key = llm_config.get('gemini_api_key')
        elif provider == 'openai':
            api_key = llm_config.get('openai_api_key')
        else:
            api_key = llm_config.get('api_key')

        # Create provider configuration
        config = LLMProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=llm_config.get('temperature', 1.0),
            max_retries=llm_config.get('max_retries', 2),
            timeout=llm_config.get('timeout'),
            extra_params=llm_config.get('extra_params', {})
        )

        return self.registry.create_provider(config)

    def create_with_fallback(
        self,
        preferred_provider: str,
        model: str,
        fallback_provider: str = "mock",
        **kwargs
    ) -> BaseLLMProvider:
        """
        Create provider with fallback support.

        Parameters
        ----------
        preferred_provider : str
            Preferred provider name
        model : str
            Model name
        fallback_provider : str, default="mock"
            Fallback provider if preferred fails
        **kwargs : Any
            Additional configuration

        Returns
        -------
        BaseLLMProvider
            Provider instance (preferred or fallback)
        """
        try:
            # Try preferred provider first
            return create_llm_provider(preferred_provider, model, **kwargs)
        except Exception as e:
            warnings.warn(
                f"Failed to create {preferred_provider} provider: {e}. "
                f"Falling back to {fallback_provider}.",
                RuntimeWarning
            )

            # Fall back to fallback provider
            fallback_model = "mock-model" if fallback_provider == "mock" else model
            return create_llm_provider(fallback_provider, fallback_model, **kwargs)

    def auto_detect_provider(self) -> BaseLLMProvider:
        """
        Auto-detect and create the best available provider.

        Returns
        -------
        BaseLLMProvider
            Best available provider instance
        """
        # Check for available API keys and create provider accordingly
        if os.getenv("GEMINI_API_KEY"):
            return create_llm_provider("google", "gemini-2.5-pro")
        elif os.getenv("OPENAI_API_KEY"):
            return create_llm_provider("openai", "gpt-4o")
        elif os.getenv("DASHSCOPE_API_KEY"):
            return create_llm_provider("tongyi", "qwen-turbo")
        elif os.getenv("ZHIPUAI_API_KEY"):
            return create_llm_provider("glm", "glm-4")
        else:
            warnings.warn(
                "No API keys found. Using mock provider for testing.",
                RuntimeWarning
            )
            return create_llm_provider("mock", "mock-model")


# Global factory instance
_llm_factory = LLMProviderFactory()


def get_llm_factory() -> LLMProviderFactory:
    """Get the global LLM provider factory."""
    return _llm_factory
