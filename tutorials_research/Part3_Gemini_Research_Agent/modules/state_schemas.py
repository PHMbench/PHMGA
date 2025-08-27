"""
Research State Schemas - Reference Architecture

Simple TypedDict-based state management for LangGraph research workflows.
Adapted from Google's Gemini LangGraph patterns for multi-provider compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field

from langgraph.graph import add_messages
import operator


# Main workflow states (simplified from reference)
class OverallState(TypedDict):
    """Main state for the research workflow - matches reference pattern"""
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add] 
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class ReflectionState(TypedDict):
    """State for reflection phase - simplified from reference"""
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    """Individual query with rationale - from reference"""
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    """State for query generation phase"""
    search_query: List[Query]


class WebSearchState(TypedDict):
    """State for individual web search execution"""
    search_query: str
    id: str


# Legacy compatibility types (for existing tutorial code)
OverallResearchState = OverallState  # Alias for backward compatibility


@dataclass(kw_only=True)
class SearchStateOutput:
    """Search state output - from reference"""
    running_summary: str = field(default=None)


# Configuration class (simplified)
class ResearchConfiguration(BaseModel):
    """Simplified configuration for multi-provider research"""
    
    # LLM settings (compatible with Part 1)
    query_generator_model: str = Field(default="default", description="Model for query generation")
    reflection_model: str = Field(default="default", description="Model for reflection")
    answer_model: str = Field(default="default", description="Model for final answer")
    
    # Research parameters
    number_of_initial_queries: int = Field(default=3, description="Number of initial queries")
    max_research_loops: int = Field(default=2, description="Maximum research loops")
    
    # Search settings 
    max_sources_per_query: int = Field(default=10, description="Max sources per query")
    parallel_search_enabled: bool = Field(default=True, description="Enable parallel search")
    search_timeout: int = Field(default=30, description="Search timeout in seconds")
    
    # Quality thresholds
    minimum_sources: int = Field(default=5, description="Minimum sources required")
    coverage_threshold: float = Field(default=0.8, description="Coverage threshold")
    academic_focus: bool = Field(default=False, description="Focus on academic sources")
    
    @classmethod
    def for_academic_research(cls) -> "ResearchConfiguration":
        """Configuration optimized for academic research"""
        return cls(
            number_of_initial_queries=4,
            max_research_loops=3,
            minimum_sources=10,
            coverage_threshold=0.9,
            academic_focus=True
        )
    
    @classmethod
    def for_quick_research(cls) -> "ResearchConfiguration":
        """Configuration for quick research"""
        return cls(
            number_of_initial_queries=2,
            max_research_loops=1,
            minimum_sources=3,
            coverage_threshold=0.6
        )


def create_initial_research_state(
    research_question: str,
    config: ResearchConfiguration,
    session_id: str = None
) -> OverallState:
    """
    Create initial state for research workflow.
    Simplified to match reference patterns.
    
    Args:
        research_question: The research question to investigate
        config: Research configuration
        session_id: Optional session identifier
        
    Returns:
        Initial OverallState
    """
    import uuid
    from langchain_core.messages import HumanMessage
    
    if session_id is None:
        session_id = f"research_{uuid.uuid4().hex[:8]}"
    
    return OverallState(
        messages=[HumanMessage(content=research_question)],
        search_query=[],
        web_research_result=[],
        sources_gathered=[],
        initial_search_query_count=config.number_of_initial_queries,
        max_research_loops=config.max_research_loops,
        research_loop_count=0,
        reasoning_model=config.answer_model
    )


# Legacy compatibility functions
def validate_research_state(state: OverallState) -> List[str]:
    """Validate research state - simplified"""
    errors = []
    
    if not state.get("messages"):
        errors.append("Missing messages")
    
    if state.get("research_loop_count", 0) < 0:
        errors.append("Invalid research loop count")
    
    if state.get("max_research_loops", 0) <= 0:
        errors.append("Invalid max research loops")
    
    return errors


if __name__ == "__main__":
    # Demonstration of simplified schema usage
    print("🔬 SIMPLIFIED RESEARCH STATE SCHEMAS")
    print("=" * 40)
    
    # Create sample configuration
    config = ResearchConfiguration.for_academic_research()
    print(f"Academic Config: {config.number_of_initial_queries} queries, {config.max_research_loops} loops")
    
    # Create initial state
    state = create_initial_research_state(
        "What are recent advances in quantum error correction?",
        config
    )
    print(f"\nInitial State Created:")
    print(f"Messages: {len(state['messages'])}")
    print(f"Research Question: {state['messages'][0].content}")
    print(f"Max Loops: {state['max_research_loops']}")
    
    # Validate state
    errors = validate_research_state(state)
    print(f"\nValidation: {'✅ Valid' if not errors else f'❌ Errors: {errors}'}")
    
    print("\n✅ Simplified schema system ready!")