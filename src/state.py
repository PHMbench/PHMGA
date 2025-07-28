from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated
from .tools.schemas import PHMOperator


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report

class PHMState(TypedDict):
    """Graph state for PHM system."""

    user_instruction: str
    reference_signal: list
    test_signal: list

    initial_plan: list[dict]
    current_plan: list[PHMOperator]
    needs_deep_research: bool
    iteration_count: int

    processed_signals: Annotated[list, operator.add]
    extracted_features: Annotated[list, operator.add]
    analysis_summary: str | None
    bug_report: dict | None
    reflection_history: list

    final_markdown_report: str
    latex_sections: dict
    final_latex_report: str | None
