"""
Simplified Research Agent Modules - Fixed Following Google Reference Architecture

Clean module structure with fixed authentication following Google's reference patterns.
All LLM creation is now done directly in graph.py with proper API key usage.
"""

# Core components (following Google reference exactly)
from .graph import conduct_research, graph
from .state import OverallState, ReflectionState, QueryGenerationState, WebSearchState
from .configuration import Configuration
from .tools_and_schemas import SearchQueryList, Reflection
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from .utils import get_research_topic, create_mock_citations, format_research_summary
from .llm_providers import check_provider_status, get_available_providers, validate_provider_setup

__all__ = [
    # Main functions
    "conduct_research",
    "graph",
    
    # State management
    "OverallState",
    "ReflectionState", 
    "QueryGenerationState",
    "WebSearchState",
    
    # Configuration
    "Configuration",
    
    # Schemas
    "SearchQueryList",
    "Reflection",
    
    # Prompts
    "get_current_date",
    "query_writer_instructions",
    "web_searcher_instructions", 
    "reflection_instructions",
    "answer_instructions",
    
    # Utilities
    "get_research_topic",
    "create_mock_citations",
    "format_research_summary",
    
    # LLM provider utilities (for status checking only)
    "check_provider_status",
    "get_available_providers", 
    "validate_provider_setup",
]