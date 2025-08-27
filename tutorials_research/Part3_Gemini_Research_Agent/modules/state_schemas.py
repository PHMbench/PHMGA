"""
Research State Schemas

Data structures for reflection-based research workflows.
Adapted from Gemini example to work with multi-provider LLM setup.
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field
from dataclasses import dataclass
import operator
from datetime import datetime


# Core schemas for structured LLM output
class SearchQuery(BaseModel):
    """Individual search query with rationale"""
    query: str = Field(description="The search query string")
    rationale: str = Field(description="Why this query is relevant to the research topic")
    priority: int = Field(default=1, description="Query priority (1=highest, 5=lowest)")


class SearchQueryList(BaseModel):
    """Collection of search queries for initial research"""
    queries: List[str] = Field(
        description="A list of search queries to be used for web research"
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic"
    )
    research_strategy: str = Field(
        description="The overall strategy behind these query choices"
    )


class ReflectionResult(BaseModel):
    """Result of reflection on research completeness"""
    is_sufficient: bool = Field(
        description="Whether the provided research is sufficient to answer the user's question"
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification"
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence in the sufficiency assessment (0.0-1.0)"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Specific aspects of the topic that need more coverage"
    )


class ResearchSource(BaseModel):
    """Individual research source with metadata"""
    url: str
    title: str
    snippet: str
    source_type: str = "web"  # web, academic, news, etc.
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    publication_date: Optional[str] = None
    authors: List[str] = Field(default_factory=list)


@dataclass
class ResearchIteration:
    """Single iteration of the research loop"""
    iteration_number: int
    queries: List[str]
    sources_found: int
    key_findings: List[str]
    knowledge_gaps: List[str]
    execution_time: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# Main state schemas for LangGraph workflow
class OverallResearchState(TypedDict):
    """
    Main state for the research workflow.
    Tracks the complete research process from question to final answer.
    """
    # Core research information
    research_question: str
    messages: Annotated[List[Dict[str, str]], operator.add]
    
    # Query management
    search_queries: Annotated[List[str], operator.add]
    initial_search_query_count: int
    current_query_batch: List[str]
    
    # Research results
    web_research_results: Annotated[List[str], operator.add]
    sources_gathered: Annotated[List[ResearchSource], operator.add]
    all_findings: Annotated[List[str], operator.add]
    
    # Reflection and iteration
    research_loop_count: int
    max_research_loops: int
    reflection_history: Annotated[List[ReflectionResult], operator.add]
    
    # Research quality metrics
    knowledge_coverage_score: float
    source_diversity_score: float
    research_depth_score: float
    
    # Configuration
    llm_provider: str
    llm_model: str
    enable_parallel_search: bool
    max_sources_per_query: int
    
    # Session tracking
    session_id: str
    start_time: str
    total_execution_time: float


class QueryGenerationState(TypedDict):
    """State for query generation phase"""
    research_question: str
    generated_queries: List[SearchQuery]
    query_generation_strategy: str
    estimated_coverage: float


class WebSearchState(TypedDict):
    """State for individual web search execution"""
    search_query: str
    query_id: str
    sources_found: List[ResearchSource]
    search_execution_time: float
    search_success: bool
    error_message: Optional[str]


class ReflectionState(TypedDict):
    """State for reflection phase"""
    current_findings: List[str]
    reflection_result: ReflectionResult
    suggested_follow_ups: List[str]
    research_completeness_score: float
    identified_gaps: List[str]


class SynthesisState(TypedDict):
    """State for final answer synthesis"""
    all_research_results: List[str]
    synthesized_answer: str
    source_citations: List[str]
    research_summary: str
    confidence_in_answer: float


# Helper classes for research management
@dataclass
class ResearchSession:
    """Complete research session tracking"""
    session_id: str
    research_question: str
    iterations: List[ResearchIteration]
    final_answer: str
    total_sources: int
    total_execution_time: float
    research_quality_score: float
    
    def add_iteration(self, iteration: ResearchIteration):
        """Add a new research iteration"""
        self.iterations.append(iteration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get research session summary"""
        return {
            "session_id": self.session_id,
            "research_question": self.research_question,
            "iterations_completed": len(self.iterations),
            "total_sources": self.total_sources,
            "total_time": self.total_execution_time,
            "quality_score": self.research_quality_score,
            "final_answer_preview": self.final_answer[:200] + "..." if len(self.final_answer) > 200 else self.final_answer
        }


class ResearchConfiguration(BaseModel):
    """Configuration for research workflow"""
    # LLM settings
    llm_provider: str = "auto"  # Uses Part 1's auto-selection
    query_model: str = "default"  # Use provider default
    reflection_model: str = "default"
    synthesis_model: str = "default"
    
    # Research parameters
    initial_query_count: int = 3
    max_research_loops: int = 2
    max_sources_per_query: int = 10
    parallel_search_enabled: bool = True
    
    # Quality thresholds
    minimum_sources: int = 5
    coverage_threshold: float = 0.8
    confidence_threshold: float = 0.7
    
    # Search configuration
    search_timeout: int = 30
    enable_source_validation: bool = True
    academic_focus: bool = False
    
    @classmethod
    def for_academic_research(cls) -> "ResearchConfiguration":
        """Configuration optimized for academic research"""
        return cls(
            initial_query_count=4,
            max_research_loops=3,
            minimum_sources=10,
            coverage_threshold=0.9,
            academic_focus=True,
            enable_source_validation=True
        )
    
    @classmethod
    def for_quick_research(cls) -> "ResearchConfiguration":
        """Configuration for quick research tasks"""
        return cls(
            initial_query_count=2,
            max_research_loops=1,
            minimum_sources=3,
            coverage_threshold=0.6,
            search_timeout=15
        )


def create_initial_research_state(
    research_question: str,
    config: ResearchConfiguration,
    session_id: str = None
) -> OverallResearchState:
    """
    Create initial state for research workflow.
    
    Args:
        research_question: The research question to investigate
        config: Research configuration
        session_id: Optional session identifier
        
    Returns:
        Initial OverallResearchState
    """
    import uuid
    from datetime import datetime
    
    if session_id is None:
        session_id = f"research_{uuid.uuid4().hex[:8]}"
    
    return OverallResearchState(
        # Core research
        research_question=research_question,
        messages=[],
        
        # Queries
        search_queries=[],
        initial_search_query_count=config.initial_query_count,
        current_query_batch=[],
        
        # Results
        web_research_results=[],
        sources_gathered=[],
        all_findings=[],
        
        # Iteration
        research_loop_count=0,
        max_research_loops=config.max_research_loops,
        reflection_history=[],
        
        # Quality
        knowledge_coverage_score=0.0,
        source_diversity_score=0.0,
        research_depth_score=0.0,
        
        # Configuration
        llm_provider=config.llm_provider,
        llm_model=config.query_model,
        enable_parallel_search=config.parallel_search_enabled,
        max_sources_per_query=config.max_sources_per_query,
        
        # Session
        session_id=session_id,
        start_time=datetime.now().isoformat(),
        total_execution_time=0.0
    )


def validate_research_state(state: OverallResearchState) -> List[str]:
    """
    Validate research state for consistency and completeness.
    
    Args:
        state: Research state to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not state.get("research_question"):
        errors.append("Missing research question")
    
    if state.get("research_loop_count", 0) < 0:
        errors.append("Invalid research loop count")
    
    if state.get("max_research_loops", 0) <= 0:
        errors.append("Invalid max research loops")
    
    # Check consistency
    if len(state.get("search_queries", [])) > 0 and len(state.get("web_research_results", [])) == 0:
        errors.append("Queries generated but no research results")
    
    # Check quality scores
    for score_field in ["knowledge_coverage_score", "source_diversity_score", "research_depth_score"]:
        score = state.get(score_field, 0.0)
        if not 0.0 <= score <= 1.0:
            errors.append(f"Invalid {score_field}: {score}")
    
    return errors


if __name__ == "__main__":
    # Demonstration of schema usage
    print("🔬 RESEARCH STATE SCHEMAS DEMONSTRATION")
    print("=" * 40)
    
    # Create sample configuration
    config = ResearchConfiguration.for_academic_research()
    print(f"Academic Config: {config.initial_query_count} queries, {config.max_research_loops} loops")
    
    # Create initial state
    state = create_initial_research_state(
        "What are recent advances in quantum error correction?",
        config
    )
    print(f"\\nInitial State Created:")
    print(f"Session ID: {state['session_id']}")
    print(f"Research Question: {state['research_question']}")
    print(f"Max Loops: {state['max_research_loops']}")
    
    # Validate state
    errors = validate_research_state(state)
    print(f"\\nValidation: {'✅ Valid' if not errors else f'❌ Errors: {errors}'}")
    
    # Example structured output
    sample_queries = SearchQueryList(
        queries=[
            "quantum error correction codes 2024",
            "fault-tolerant quantum computing recent advances", 
            "NISQ device error mitigation techniques"
        ],
        rationale="These queries cover different aspects of quantum error correction",
        research_strategy="Multi-perspective approach covering theory, applications, and current limitations"
    )
    
    print(f"\\n📋 Sample Query Generation:")
    print(f"Queries: {len(sample_queries.queries)}")
    print(f"Strategy: {sample_queries.research_strategy}")
    
    print("\\n✅ Schema system ready for research workflows!")