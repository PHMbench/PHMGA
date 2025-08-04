from __future__ import annotations

from langgraph.graph import StateGraph, END, START

from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.execute_agent import execute_agent
from .agents.inquirer_agent import inquirer_agent
from .agents.plan_agent import plan_agent
from .agents.reflect_agent import reflect_agent_node
from .agents.report_agent import report_agent_node
from .agents.shallow_ml_agent import shallow_ml_agent
from .states.phm_states import PHMState


def build_builder_graph() -> StateGraph:
    """
    Constructs the graph responsible for iteratively building the computational DAG.
    This graph uses a plan-execute-reflect loop to generate a valid DAG.
    """
    builder = StateGraph(PHMState)

    builder.add_node("plan", plan_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE"))

    builder.set_entry_point("plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")

    # The reflection step decides whether to loop back to planning or to finish.
    builder.add_conditional_edges(
        "reflect",
        # MODIFIED: Use direct attribute access instead of .get() for Pydantic models
        lambda state: "revise" if state.needs_revision else END,
        {
            "revise": "plan",
            END: END,
        },
    )
    
    return builder.compile()


def build_executor_graph() -> StateGraph:
    """
    Constructs the graph that executes a finalized computational DAG.
    This graph performs similarity analysis, dataset preparation, model training, and reporting.
    """
    builder = StateGraph(PHMState)

    # Define the nodes for the execution pipeline
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine", "euclidean"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", lambda state: {"ml_results": shallow_ml_agent(datasets=state.datasets)})
    builder.add_node("report", report_agent_node)

    # Define the linear flow of the execution graph
    builder.set_entry_point("inquire")
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)

    return builder.compile()


