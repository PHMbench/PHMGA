from __future__ import annotations

import logging
from typing import Callable

from langgraph.graph import StateGraph, END, START

from .state import PHMState
from .configuration import Config
from .agents import (
    planner,
    dispatcher,
    process_signals,
    extract_features,
    analyze,
    reflect,
    write_report,
)
from .database import SQLiteCheckpointer

__all__ = ["build_graph"]


def _route_reflection(state: PHMState) -> str:
    if state["is_sufficient"]:
        return "report"
    return "dispatcher"


def build_graph(config: Config, db_path: str = "phm.db"):
    """Build the PHM LangGraph."""
    logging.info("Building graph")
    builder = StateGraph(PHMState)

    builder.add_node("planner", lambda s: planner(s, config))
    builder.add_node("dispatcher", dispatcher)
    builder.add_node("signal_processing", lambda s: process_signals(s, config))
    builder.add_node("feature_extraction", extract_features)
    builder.add_node("analysis", analyze)
    builder.add_node("reflection", lambda s: reflect(s, config))
    builder.add_node("report", write_report)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "dispatcher")
    builder.add_edge("dispatcher", "signal_processing")
    builder.add_edge("signal_processing", "feature_extraction")
    builder.add_edge("feature_extraction", "analysis")
    builder.add_edge("analysis", "reflection")

    builder.add_conditional_edges("reflection", _route_reflection, {
        "dispatcher": "dispatcher",
        "report": "report",
    })
    builder.add_edge("report", END)

    checkpointer = SQLiteCheckpointer(db_path)
    return builder.compile(checkpointer=checkpointer.saver)
