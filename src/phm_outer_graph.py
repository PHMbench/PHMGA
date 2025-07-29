from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver as SqliteCheckpointer

from .states.phm_states import PHMState
from .agents.plan_agent import plan_agent
from .agents.execute_agent import execute_agent
from .agents.reflect_agent import reflect_agent
from .agents.report_agent import report_agent


def build_outer_graph() -> StateGraph:
    """Construct the static outer workflow graph.

    Returns
    -------
    StateGraph
        Compiled graph controlling the plan/execute/reflect/report loop.
    """
    builder = StateGraph(PHMState)

    builder.add_node("plan", plan_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("reflect", reflect_agent)
    builder.add_node("report", report_agent)

    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")

    builder.add_conditional_edges(
        "reflect",
        lambda state: "revise" if state.needs_revision else "done",
        {
            "done": "report",
            "revise": "plan",
        },
    )
    builder.set_finish_point("report")

    # Checkpointer is optional and can cause issues if not correctly managed.
    # For this simplified test workflow we compile without a persistent
    # checkpointer to avoid connection handling errors.
    return builder.compile()

