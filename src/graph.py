"""Definition of the PHM LangGraph."""

from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from .agents.analysis import analyze
from .agents.dispatcher import dispatcher
from .agents.feature_extraction import feature_extractor
from .agents.planner import planner
from .agents.reflection import reflect
from .agents.report_writer import write_report
from .agents.signal_processing import signal_processor
from .configuration import Config
from .database import SQLiteCheckpointer
from .state import PHMState
from langgraph.graph.state import CompiledStateGraph

__all__ = ["build_graph"]


from typing import Any


def build_graph(config: Config, db_path: str = "phm.sqlite") -> CompiledStateGraph[PHMState, Any, Any]:
    builder = StateGraph(PHMState)
    builder.add_node("planner", planner)
    builder.add_node("dispatcher", dispatcher)
    builder.add_node("signal_processing", lambda s: signal_processor(s, config))
    builder.add_node("feature_extraction", feature_extractor)
    builder.add_node("analysis", analyze)
    builder.add_node("reflection", reflect)
    builder.add_node("report", write_report)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_edge("dispatcher", "signal_processing")
    builder.add_edge("signal_processing", "feature_extraction")
    builder.add_edge("feature_extraction", "analysis")
    builder.add_edge("analysis", "reflection")

    def decide(state: PHMState) -> str:
        if state["is_sufficient"] or state["iteration_count"] >= config.max_loops:
            return "report"
        return "dispatcher"

    builder.add_conditional_edges("reflection", decide, ["dispatcher", "report"])
    builder.add_edge("report", END)

    checkpointer = SQLiteCheckpointer(db_path)
    return builder.compile(checkpointer=checkpointer)


