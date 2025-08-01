from __future__ import annotations

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver 

from .states.phm_states import PHMState
from .agents.plan_agent import plan_agent
from .agents.execute_agent import execute_agent
from .agents.inquirer_agent import inquirer_agent
from .agents.reflect_agent import reflect_agent
from .agents.report_agent import report_agent
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
    builder.add_node("execute_pre", execute_agent)
    builder.add_node("reflect_dag", lambda state: reflect_agent(state, stage="DAG_REVIEW"))
    builder.add_node("inquire", inquirer_agent)
    builder.add_node("execute_post", execute_agent)
    builder.add_node("reflect_ins", lambda state: reflect_agent(state, stage="INSIGHTS_REVIEW"))
    builder.add_node("report", report_agent)

    # builder.add_edge(START, "plan")
    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute_pre")
    builder.add_edge("execute_pre", "reflect_dag")
    builder.add_edge("reflect_dag", "inquire")
    builder.add_edge("inquire", "execute_post")
    builder.add_edge("execute_post", "reflect_ins")

    builder.add_conditional_edges(
        "reflect_ins",
        lambda state: "revise" if state.needs_revision else "done",
        {
            "done": "report",
            "revise": "plan",
        },
    )
    builder.set_finish_point("report")
    # builder.add_edge("report", END)
    import os
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/phm_agent.db", check_same_thread=False)
    # Here is our checkpointer 

    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)


