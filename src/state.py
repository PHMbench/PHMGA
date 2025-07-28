from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


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


# === PHM System Models ===
import uuid
from typing import List, Dict, Any, Annotated
from pydantic import BaseModel, Field


class SignalData(BaseModel):
    """Represents a batch of raw input signals."""

    signal_id: str = Field(default_factory=lambda: f"sig_{uuid.uuid4().hex[:8]}")
    data: List[List[List[float]]]
    sampling_rate: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedSignal(BaseModel):
    """Output of a single signal processing method."""

    processed_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:8]}")
    source_signal_id: str
    method: str
    processed_data: Any


class ExtractedFeatures(BaseModel):
    """Feature set extracted from a batch of signals."""

    feature_set_id: str = Field(default_factory=lambda: f"feat_{uuid.uuid4().hex[:8]}")
    source_processed_id: str
    features: List[Dict[str, float]]


class AnalysisInsight(BaseModel):
    """Concrete insight produced by analysis or reflection."""

    insight_id: str = Field(default_factory=lambda: f"ins_{uuid.uuid4().hex[:8]}")
    content: str
    severity_score: float = Field(ge=0.0, le=1.0)
    supporting_feature_ids: List[str]


class PHMState(TypedDict):
    """Central state for the PHM LangGraph pipeline."""

    user_instruction: str
    reference_signal: SignalData
    test_signal: SignalData

    plan: Dict[str, Any]
    reflection_history: List[str]
    is_sufficient: bool
    iteration_count: int

    processed_signals: Annotated[List[ProcessedSignal], operator.add]
    extracted_features: Annotated[List[ExtractedFeatures], operator.add]

    analysis_results: List[AnalysisInsight]
    final_decision: str

    final_report: str


__all__ = [
    "OverallState",
    "ReflectionState",
    "Query",
    "QueryGenerationState",
    "WebSearchState",
    "SearchStateOutput",
    "SignalData",
    "ProcessedSignal",
    "ExtractedFeatures",
    "AnalysisInsight",
    "PHMState",
]
