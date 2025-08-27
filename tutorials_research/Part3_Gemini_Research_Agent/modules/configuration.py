"""
Configuration for Multi-Provider Research Workflows

Configuration system adapted from Google's reference architecture to support
multi-provider LLM systems while maintaining compatibility with LangGraph patterns.
"""

import os
import sys
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

# Add path for Part 1 LLM providers
sys.path.append('../Part1_Foundations/modules')


class Configuration(BaseModel):
    """
    Multi-provider configuration for research agent workflows.
    
    Adapts Google's reference configuration to support multiple LLM providers
    while maintaining the same interface for LangGraph compatibility.
    """
    
    # LLM Provider settings (adapted for multi-provider support)
    llm_provider: str = Field(
        default="auto",
        metadata={
            "description": "LLM provider to use: auto, google, openai, dashscope, zhipuai"
        }
    )
    
    query_generator_model: str = Field(
        default="default",
        metadata={
            "description": "Model for query generation (provider-specific default if 'default')"
        }
    )
    
    reflection_model: str = Field(
        default="default", 
        metadata={
            "description": "Model for reflection analysis (provider-specific default if 'default')"
        }
    )
    
    answer_model: str = Field(
        default="default",
        metadata={
            "description": "Model for final answer synthesis (provider-specific default if 'default')"
        }
    )
    
    # Research workflow parameters (from reference)
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "Number of initial search queries to generate"}
    )
    
    max_research_loops: int = Field(
        default=2,
        metadata={"description": "Maximum number of research loops to perform"}
    )
    
    # Search and quality settings
    max_sources_per_query: int = Field(
        default=5,
        metadata={"description": "Maximum sources to collect per search query"}
    )
    
    search_timeout: int = Field(
        default=30,
        metadata={"description": "Search timeout in seconds"}
    )
    
    minimum_sources: int = Field(
        default=5,
        metadata={"description": "Minimum sources required for completion"}
    )
    
    coverage_threshold: float = Field(
        default=0.8,
        metadata={"description": "Coverage threshold for research completeness"}
    )
    
    # Multi-provider specific settings
    temperature: float = Field(
        default=0.7,
        metadata={"description": "Temperature for LLM responses"}
    )
    
    use_fast_mode: bool = Field(
        default=False,
        metadata={"description": "Use faster/cheaper model variants when available"}
    )
    
    google_search_enabled: bool = Field(
        default=None,
        metadata={"description": "Enable Google native search (auto-detect if None)"}
    )
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """
        Create Configuration from RunnableConfig (reference pattern).
        
        Supports both environment variables and runtime configuration.
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        
        # Get values from environment or config
        raw_values: dict[str, Any] = {}
        
        for field_name, field_info in cls.model_fields.items():
            # Try config first, then environment, then default
            env_name = field_name.upper()
            value = configurable.get(field_name) or os.environ.get(env_name)
            
            if value is not None:
                # Convert string values to appropriate types
                field_type = field_info.annotation
                if field_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                elif field_type == int:
                    value = int(value)
                elif field_type == float:
                    value = float(value)
                
                raw_values[field_name] = value
        
        return cls(**raw_values)
    
    @classmethod
    def for_academic_research(cls) -> "Configuration":
        """Configuration optimized for academic research"""
        return cls(
            number_of_initial_queries=4,
            max_research_loops=3,
            minimum_sources=8,
            coverage_threshold=0.9,
            temperature=0.3,  # Lower for more focused academic research
            search_timeout=45
        )
    
    @classmethod 
    def for_quick_research(cls) -> "Configuration":
        """Configuration optimized for quick research tasks"""
        return cls(
            number_of_initial_queries=2,
            max_research_loops=1,
            minimum_sources=3,
            coverage_threshold=0.6,
            use_fast_mode=True,
            search_timeout=15
        )
    
    @classmethod
    def for_comprehensive_research(cls) -> "Configuration":
        """Configuration for comprehensive, in-depth research"""
        return cls(
            number_of_initial_queries=5,
            max_research_loops=4,
            minimum_sources=12,
            coverage_threshold=0.95,
            temperature=0.5,
            search_timeout=60
        )
    
    def get_llm_config(self) -> dict[str, Any]:
        """
        Get LLM configuration for Part 1 provider system.
        
        Returns configuration dict suitable for create_research_llm().
        """
        config = {
            "temperature": self.temperature,
            "fast_mode": self.use_fast_mode
        }
        
        # Add provider if specified
        if self.llm_provider and self.llm_provider != "auto":
            config["provider_name"] = self.llm_provider
        
        return config
    
    def get_model_for_task(self, task: str) -> str:
        """
        Get specific model name for a task.
        
        Args:
            task: Task type ("query", "reflection", "answer")
            
        Returns:
            Model name or "default" for provider default
        """
        model_mapping = {
            "query": self.query_generator_model,
            "reflection": self.reflection_model, 
            "answer": self.answer_model
        }
        
        return model_mapping.get(task, "default")
    
    def is_google_search_available(self) -> bool:
        """Check if Google search is available and enabled"""
        
        if self.google_search_enabled is False:
            return False
        
        # Auto-detect if not explicitly disabled
        google_api_key = os.getenv('GEMINI_API_KEY')
        google_search_key = os.getenv('GOOGLE_API_KEY')
        google_search_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        return all([google_api_key, google_search_key, google_search_id])
    
    def get_search_strategy(self) -> str:
        """Determine search strategy based on configuration and availability"""
        
        if self.is_google_search_available():
            return "google_native"
        else:
            return "simulated"
    
    def validate_configuration(self) -> list[str]:
        """Validate configuration and return any warnings or errors"""
        
        warnings = []
        
        # Check for basic configuration issues
        if self.number_of_initial_queries < 1:
            warnings.append("number_of_initial_queries should be at least 1")
        
        if self.max_research_loops < 1:
            warnings.append("max_research_loops should be at least 1")
        
        if self.coverage_threshold < 0.0 or self.coverage_threshold > 1.0:
            warnings.append("coverage_threshold should be between 0.0 and 1.0")
        
        if self.temperature < 0.0 or self.temperature > 2.0:
            warnings.append("temperature should be between 0.0 and 2.0")
        
        # Check provider availability
        try:
            from llm_providers import ResearchLLMFactory
            factory = ResearchLLMFactory()
            available_providers = factory.get_available_providers()
            
            if not any(p["available"] for p in available_providers.values()):
                warnings.append("No LLM providers are available - check API keys")
            
        except ImportError:
            warnings.append("Cannot validate LLM provider availability - Part 1 modules not found")
        
        return warnings
    
    def get_summary(self) -> dict[str, Any]:
        """Get configuration summary for logging/display"""
        
        return {
            "llm_provider": self.llm_provider,
            "initial_queries": self.number_of_initial_queries,
            "max_loops": self.max_research_loops,
            "search_strategy": self.get_search_strategy(),
            "temperature": self.temperature,
            "fast_mode": self.use_fast_mode,
            "coverage_threshold": self.coverage_threshold
        }


def create_default_configuration() -> Configuration:
    """Create default configuration with automatic provider detection"""
    
    config = Configuration()
    
    # Auto-detect best available provider
    try:
        from llm_providers import ResearchLLMFactory
        factory = ResearchLLMFactory()
        recommended = factory.get_recommended_provider()
        
        if recommended:
            config.llm_provider = recommended.value
        
    except ImportError:
        # Fallback if Part 1 not available
        config.llm_provider = "auto"
    
    return config


def load_configuration_from_env() -> Configuration:
    """Load configuration from environment variables"""
    
    return Configuration.from_runnable_config(None)


if __name__ == "__main__":
    print("⚙️ MULTI-PROVIDER RESEARCH CONFIGURATION")
    print("=" * 42)
    
    # Demonstrate configuration creation
    print("\n📋 Configuration Profiles:")
    profiles = [
        ("Default", Configuration()),
        ("Academic", Configuration.for_academic_research()),
        ("Quick", Configuration.for_quick_research()),
        ("Comprehensive", Configuration.for_comprehensive_research())
    ]
    
    for name, config in profiles:
        summary = config.get_summary()
        print(f"\n{name} Profile:")
        for key, value in summary.items():
            print(f"   • {key}: {value}")
    
    print("\n🔍 Environment Detection:")
    default_config = create_default_configuration()
    search_strategy = default_config.get_search_strategy()
    print(f"   • Search Strategy: {search_strategy}")
    print(f"   • LLM Provider: {default_config.llm_provider}")
    
    # Validate configuration
    warnings = default_config.validate_configuration()
    if warnings:
        print(f"\n⚠️ Configuration Warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    else:
        print(f"\n✅ Configuration validated successfully")
    
    print("\n🔧 Usage Examples:")
    print("""
    # Create configuration
    config = Configuration.for_academic_research()
    
    # Use with LangGraph
    app = graph.compile(config_schema=Configuration)
    
    # Get LLM config for Part 1 providers
    llm_config = config.get_llm_config()
    llm = create_research_llm(**llm_config)
    
    # Load from environment
    config = load_configuration_from_env()
    """)
    
    print("\n✅ Multi-provider configuration system ready!")