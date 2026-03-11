from __future__ import annotations

from typing import Any

from src.agents.execute_agent import execute_agent
from src.agents.plan_agent import plan_agent
from src.agents.reflect_agent import reflect_agent_node
from src.states.phm_states import PHMState

from .common import END, LANGGRAPH_OK, FallbackGraph, StateGraph, run_node


def build_builder_graph() -> Any:
    def _should_continue(state: PHMState) -> str:
        if not bool(getattr(state, "needs_revision", False)):
            return END
        iteration_count = int(getattr(state, "iteration_count", 0) or 0)
        max_iterations = int(getattr(state, "max_builder_iterations", 50) or 50)
        return "plan" if iteration_count < max_iterations else END

    if not LANGGRAPH_OK:  # pragma: no cover
        return FallbackGraph(
            [
                ("plan", plan_agent),
                ("execute", execute_agent),
                ("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE")),
            ]
        )

    builder = StateGraph(PHMState)  # type: ignore[misc]
    builder.add_node("plan", lambda state: run_node("plan", plan_agent, state))
    builder.add_node("execute", lambda state: run_node("execute", execute_agent, state))
    builder.add_node(
        "reflect",
        lambda state: run_node("reflect", lambda s: reflect_agent_node(s, stage="POST_EXECUTE"), state),
    )
    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")
    builder.add_conditional_edges(
        "reflect",
        _should_continue,
        {
            "plan": "plan",
            END: END,
        },
    )
    return builder.compile()


def build_outer_graph() -> Any:  # pragma: no cover
    return build_builder_graph()
