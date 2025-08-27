"""
Tools and Schemas - Following Google Reference Architecture

Simple Pydantic models matching the reference implementation
from Google's Gemini LangGraph quickstart.
"""

from typing import List
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    """Search query list for structured output - matches reference exactly"""
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    """Reflection model for structured output - matches reference exactly"""
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )