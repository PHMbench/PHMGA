"""
Tools and Schemas for Research Workflows

Pydantic models for structured LLM output in research agents.
Adapted from Google's Gemini LangGraph reference implementation.
"""

from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    """
    Structured output for query generation.
    Used with .with_structured_output() for reliable JSON parsing.
    """
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    """
    Structured output for research reflection and gap analysis.
    Determines whether to continue research or finalize answer.
    """
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


# Additional schemas for multi-provider compatibility
class ResearchSource(BaseModel):
    """Individual research source with metadata"""
    url: str = Field(description="Source URL")
    title: str = Field(description="Source title") 
    snippet: str = Field(description="Source snippet or summary")
    source_type: str = Field(default="web", description="Type of source (web, academic, news)")
    relevance_score: float = Field(default=0.0, description="Relevance score 0-1")
    credibility_score: float = Field(default=0.0, description="Credibility score 0-1")


class SearchResult(BaseModel):
    """Structured search result for multi-provider search"""
    sources: List[ResearchSource] = Field(description="List of research sources found")
    query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results found")
    search_time: float = Field(description="Time taken for search in seconds")


class QueryGenerationResult(BaseModel):
    """Result of query generation process"""
    queries: List[str] = Field(description="Generated search queries")
    rationale: str = Field(description="Explanation of query strategy")
    research_strategy: str = Field(description="Overall research approach")


class ReflectionAnalysis(BaseModel):
    """Extended reflection analysis for research completeness"""
    is_sufficient: bool = Field(description="Whether research is sufficient")
    knowledge_gap: str = Field(description="Description of knowledge gaps")
    follow_up_queries: List[str] = Field(description="Follow-up queries to address gaps")
    confidence_score: float = Field(
        default=0.0, 
        description="Confidence in sufficiency assessment (0.0-1.0)"
    )
    missing_aspects: List[str] = Field(
        default_factory=list,
        description="Specific aspects needing more coverage"
    )


if __name__ == "__main__":
    # Demonstration of structured output schemas
    print("🛠️ RESEARCH TOOLS AND SCHEMAS")
    print("=" * 35)
    
    # Example SearchQueryList
    sample_queries = SearchQueryList(
        query=[
            "quantum error correction codes 2024",
            "fault-tolerant quantum computing recent advances",
            "NISQ device error mitigation techniques"
        ],
        rationale="These queries cover different aspects of quantum error correction: theory, applications, and current limitations."
    )
    
    print("📋 Sample Query Generation:")
    print(f"Queries: {len(sample_queries.query)}")
    print(f"Rationale: {sample_queries.rationale[:50]}...")
    
    # Example Reflection
    sample_reflection = Reflection(
        is_sufficient=False,
        knowledge_gap="Missing information about recent experimental results and commercial applications.",
        follow_up_queries=[
            "quantum error correction experimental demonstrations 2024",
            "commercial quantum computing error correction applications"
        ]
    )
    
    print(f"\n🔍 Sample Reflection:")
    print(f"Sufficient: {sample_reflection.is_sufficient}")
    print(f"Gap: {sample_reflection.knowledge_gap[:50]}...")
    print(f"Follow-ups: {len(sample_reflection.follow_up_queries)}")
    
    print(f"\n✅ Structured output schemas ready for LLM integration!")