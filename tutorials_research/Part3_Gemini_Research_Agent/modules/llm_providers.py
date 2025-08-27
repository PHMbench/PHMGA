"""
LLM Providers - Utility Functions for Provider Status

Simple utility functions for checking LLM provider availability.
The main LLM creation is now done directly in graph.py following Google's reference pattern.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def check_provider_status():
    """Check status of all providers for debugging - matches the tutorial interface"""
    providers = {
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY", 
        "dashscope": "DASHSCOPE_API_KEY",
        "zhipuai": "ZHIPUAI_API_KEY"
    }
    
    print("🔍 LLM Provider Status:")
    available_providers = []
    
    for provider, env_key in providers.items():
        api_key = os.getenv(env_key)
        if api_key:
            status = "✅ Available"
            available_providers.append(provider)
        else:
            status = "❌ Not configured"
        print(f"   {provider.upper():12} {status}")
    
    if available_providers:
        print(f"🎯 Available providers: {', '.join(available_providers)}")
        return available_providers[0]  # Return first available for auto-selection
    else:
        print("⚠️ No providers available - configure API keys")
        print("\n💡 To configure API keys:")
        print("   export GEMINI_API_KEY='your_gemini_key'")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export DASHSCOPE_API_KEY='your_dashscope_key'")
        print("   export ZHIPUAI_API_KEY='your_zhipuai_key'")
        return None


def get_available_providers():
    """Get list of available providers based on API key configuration"""
    providers = []
    
    if os.getenv("GEMINI_API_KEY"):
        providers.append("google")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("DASHSCOPE_API_KEY"):
        providers.append("dashscope")
    if os.getenv("ZHIPUAI_API_KEY"):
        providers.append("zhipuai")
    
    return providers


def validate_provider_setup(provider: str) -> bool:
    """Validate that a specific provider is properly configured"""
    env_key_map = {
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY", 
        "zhipuai": "ZHIPUAI_API_KEY"
    }
    
    env_key = env_key_map.get(provider)
    if not env_key:
        return False
    
    api_key = os.getenv(env_key)
    return bool(api_key and len(api_key.strip()) > 0)


# Legacy compatibility function (deprecated - use direct instantiation in graph.py)
def create_research_llm(*args, **kwargs):
    """
    DEPRECATED: Use direct LLM instantiation in graph.py instead.
    
    This function is kept for backward compatibility but should not be used.
    The new approach follows Google's reference pattern with direct API key usage.
    """
    raise NotImplementedError(
        "create_research_llm is deprecated. "
        "Use direct LLM instantiation in graph.py following Google's reference pattern."
    )


if __name__ == "__main__":
    print("🔧 LLM Provider Utilities")
    print("=" * 30)
    
    check_provider_status()
    
    available = get_available_providers()
    if available:
        print(f"\n✅ {len(available)} provider(s) configured")
        for provider in available:
            is_valid = validate_provider_setup(provider)
            print(f"   • {provider}: {'Valid' if is_valid else 'Invalid'}")
    else:
        print("\n❌ No providers configured")
        print("Please set up API keys to use the research system.")