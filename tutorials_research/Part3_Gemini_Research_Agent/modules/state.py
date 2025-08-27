"""
State Schemas - Following Google Reference Architecture

Simple TypedDict-based state management matching the reference implementation
from Google's Gemini LangGraph quickstart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import add_messages
import operator


# Main workflow states (Exact match with Google reference)
class OverallState(TypedDict):
    """Main state for the research workflow - matches reference exactly"""
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class ReflectionState(TypedDict):
    """State for reflection phase - matches reference exactly"""
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
    search_query: list[Query]


class WebSearchState(TypedDict):
    """State for individual web search execution"""
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    """Search state output - from reference"""
    running_summary: str = field(default=None)  # Final report