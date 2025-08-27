"""
Multi-Provider LLM Setup for Research Applications

This module provides unified access to multiple LLM providers commonly used
in academic research, with support for global accessibility.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv


class LLMProvider(Enum):
    """Supported LLM providers for research applications"""
    GOOGLE = "google"
    OPENAI = "openai" 
    DASHSCOPE = "dashscope"
    ZHIPUAI = "zhipuai"


class ResearchLLMFactory:
    """Factory for creating LLM instances for research tasks"""
    
    PROVIDER_CONFIGS = {
        LLMProvider.GOOGLE: {
            "env_key": "GEMINI_API_KEY",
            "default_model": "gemini-2.5-pro",
            "fast_model": "gemini-2.5-flash",
            "description": "Google Gemini - Excellent for mathematical reasoning"
        },
        LLMProvider.OPENAI: {
            "env_key": "OPENAI_API_KEY", 
            "default_model": "gpt-4o",
            "fast_model": "gpt-4o-mini",
            "description": "OpenAI GPT - Reliable for code understanding"
        },
        LLMProvider.DASHSCOPE: {
            "env_key": "DASHSCOPE_API_KEY",
            "default_model": "qwen-plus",
            "fast_model": "qwen-plus", # qwen-plus qwen-turbo
            "description": "DashScope Qwen - Cost-effective with good performance"
        },
        LLMProvider.ZHIPUAI: {
            "env_key": "ZHIPUAI_API_KEY",
            "default_model": "glm-4",
            "fast_model": "glm-4-flash",
            "description": "Zhipu AI GLM - Optimized for Chinese researchers"
        }
    }
    
    @classmethod
    def create_llm(cls, 
                   provider: LLMProvider,
                   model: Optional[str] = None,
                   temperature: float = 0.7,
                   max_tokens: Optional[int] = None,
                   **kwargs):
        """
        Create an LLM instance for the specified provider.
        
        Args:
            provider: LLM provider to use
            model: Specific model name (uses default if None)
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum response length
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If API key is not found
            ImportError: If required package is not installed
        """
        # Load environment variables
        load_dotenv()
        
        config = cls.PROVIDER_CONFIGS[provider]
        api_key = os.getenv(config["env_key"])
        
        if not api_key:
            raise ValueError(
                f"API key not found for {provider.value}. "
                f"Set {config['env_key']} in your environment or .env file."
            )
        
        if model is None:
            model = config["default_model"]
            
        try:
            if provider == LLMProvider.GOOGLE:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
            elif provider == LLMProvider.OPENAI:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
            elif provider == LLMProvider.DASHSCOPE:
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
            elif provider == LLMProvider.ZHIPUAI:
                from langchain_community.chat_models import ChatZhipuAI
                return ChatZhipuAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    **kwargs
                )
                
        except ImportError as e:
            package_map = {
                LLMProvider.GOOGLE: "langchain-google-genai",
                LLMProvider.OPENAI: "langchain-openai", 
                LLMProvider.DASHSCOPE: "dashscope",
                LLMProvider.ZHIPUAI: "zhipuai"
            }
            raise ImportError(
                f"Required package not installed for {provider.value}. "
                f"Install with: pip install {package_map[provider]}"
            ) from e
    
    @classmethod
    def get_available_providers(cls) -> Dict[LLMProvider, Dict[str, Any]]:
        """
        Get list of available providers based on configured API keys.
        
        Returns:
            Dictionary mapping providers to their configuration and availability
        """
        load_dotenv()
        available = {}
        
        for provider, config in cls.PROVIDER_CONFIGS.items():
            api_key = os.getenv(config["env_key"])
            available[provider] = {
                **config,
                "available": bool(api_key),
                "api_key_configured": bool(api_key)
            }
            
        return available
    
    @classmethod
    def get_recommended_provider(cls) -> Optional[LLMProvider]:
        """
        Get recommended provider based on availability and research use cases.
        
        Priority: Google (math) > OpenAI (reliability) > DashScope (cost) > Zhipu
        
        Returns:
            Recommended provider or None if none available
        """
        available = cls.get_available_providers()
        
        # Priority order for research applications
        priority_order = [
            LLMProvider.GOOGLE,    # Best for mathematical reasoning
            LLMProvider.OPENAI,    # Most reliable
            LLMProvider.DASHSCOPE, # Cost-effective
            LLMProvider.ZHIPUAI    # Regional option
        ]
        
        for provider in priority_order:
            if available[provider]["available"]:
                return provider
                
        return None


def create_research_llm(provider_name: Optional[str] = None,
                       model: Optional[str] = None,
                       temperature: float = 0.7,
                       fast_mode: bool = False,
                       **kwargs):
    """
    Convenience function to create LLM for research applications.
    
    Args:
        provider_name: Provider name (google, openai, dashscope, zhipuai)
        model: Specific model name
        temperature: Response randomness
        fast_mode: Use faster/cheaper model variant
        **kwargs: Additional parameters
        
    Returns:
        Configured LLM instance
        
    Example:
        >>> llm = create_research_llm("google", temperature=0.3)
        >>> llm = create_research_llm(fast_mode=True)  # Auto-select available provider
    """
    factory = ResearchLLMFactory()
    
    if provider_name is None:
        # Auto-select best available provider
        provider = factory.get_recommended_provider()
        if provider is None:
            raise ValueError("No LLM providers configured. Please set up API keys.")
    else:
        provider = LLMProvider(provider_name)
    
    # Use fast model if requested
    if fast_mode and model is None:
        model = factory.PROVIDER_CONFIGS[provider]["fast_model"]
    
    return factory.create_llm(
        provider=provider,
        model=model,
        temperature=temperature,
        **kwargs
    )


def list_research_providers():
    """Print available LLM providers and their status"""
    factory = ResearchLLMFactory()
    available = factory.get_available_providers()
    
    print("🔍 Available LLM Providers for Research:")
    print("-" * 60)
    
    for provider, config in available.items():
        status = "✅" if config["available"] else "❌"
        print(f"{status} {provider.value.upper():<10} - {config['description']}")
        print(f"   Default: {config['default_model']}")
        print(f"   Fast: {config['fast_model']}")
        if not config["available"]:
            print(f"   ⚠️  Set {config['env_key']} to enable")
        print()
    
    recommended = factory.get_recommended_provider()
    if recommended:
        print(f"🎯 Recommended: {recommended.value.upper()}")
    else:
        print("⚠️  No providers available - configure API keys")


if __name__ == "__main__":
    # Demo and testing
    list_research_providers()
    
    try:
        # Try to create a research LLM
        llm = create_research_llm(temperature=0.3)
        print(f"✅ Successfully created LLM: {type(llm).__name__}")
        
        # Test simple query
        response = llm.invoke("Hello! Please respond with 'Research LLM ready'")
        print(f"🧪 Test response: {response.content}")
        
    except Exception as e:
        print(f"❌ Error creating LLM: {e}")
        print("\n💡 Make sure you have:")
        print("1. Installed required packages: pip install -r requirements.txt")
        print("2. Set up API keys in .env file")
        print("3. Have internet connectivity")