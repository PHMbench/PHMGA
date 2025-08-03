from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver 

from .states.phm_states import PHMState
from .agents.plan_agent import plan_agent
from .agents.execute_agent import execute_agent
from .agents.inquirer_agent import inquirer_agent
from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.shallow_ml_agent import shallow_ml_agent
from .agents.reflect_agent import reflect_agent_node
from .agents.report_agent import report_agent_node
from langgraph.graph import START, END

import sqlite3

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
    builder.add_node("reflect", lambda state: reflect_agent_node(state, stage="POST_EXEC"))
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", lambda state: shallow_ml_agent(state.datasets))
    builder.add_node("report", report_agent_node)

    builder.add_edge(START, "plan")
    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")
    builder.add_conditional_edges(
        "reflect",
        lambda state: "revise" if state.needs_revision else "done",
        {
            "done": "inquire",
            "revise": "plan",
        },
    )
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)
    import os
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/phm_agent.db", check_same_thread=False)
    # Here is our checkpointer 

    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


