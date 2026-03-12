"""Reflection agent for validating DAG sufficiency at the workflow layer."""

from __future__ import annotations

from src.llm import OfflineLLM
from src.states import WorkflowState


def reflect_agent(state: WorkflowState, llm: OfflineLLM) -> WorkflowState:
    """Record a lightweight sufficiency judgement without mutating the DAG."""
    node_count = len(state.dag.nodes) if state.dag else 0
    state.reflection_history.append(llm.reflect(node_count, state.graph_path))
    state.status = "reflected"
    return state
