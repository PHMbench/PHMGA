from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.execute_agent import execute_agent
from .agents.inquirer_agent import inquirer_agent
from .agents.plan_agent import plan_agent
from .agents.reflect_agent import reflect_agent_node
from .agents.report_agent import report_agent_node
from .agents.shallow_ml_agent import shallow_ml_agent
from .states.phm_states import PHMState
from .utils import get_dag_depth


VALID_GRAPHS = {"builder", "executor", "with_report"}


def _builder_transition(state: PHMState):
    if state.last_reflection_decision in {"need_patch", "need_replan"} and get_dag_depth(state.dag_state) < state.max_depth:
        return "plan"
    return END


def _outer_transition(state: PHMState):
    if state.last_reflection_decision == "finish":
        return "inquire"
    if state.last_reflection_decision in {"need_patch", "need_replan"} and get_dag_depth(state.dag_state) < state.max_depth:
        return "plan"
    return END


def _train_node(state: PHMState):
    return {"ml_results": shallow_ml_agent(datasets=state.datasets)}


def build_builder_graph() -> StateGraph:
    """Build the graph responsible for iterative DAG construction."""
    builder = StateGraph(PHMState)
    builder.add_node("plan", plan_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE"))

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")
    builder.add_conditional_edges(
        "reflect",
        _builder_transition,
        {
            "plan": "plan",
            END: END,
        },
    )
    return builder.compile()


def build_executor_graph() -> StateGraph:
    """Build the graph that executes a finalized DAG and emits a report."""
    builder = StateGraph(PHMState)
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine", "euclidean"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", _train_node)
    builder.add_node("report", report_agent_node)

    builder.add_edge(START, "inquire")
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)
    return builder.compile()


def build_outer_graph() -> StateGraph:
    """Build the complete builder-to-report workflow."""
    builder = StateGraph(PHMState)
    builder.add_node("plan", plan_agent)
    builder.add_node("execute", execute_agent)
    builder.add_node("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE"))
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine", "euclidean"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", _train_node)
    builder.add_node("report", report_agent_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")
    builder.add_conditional_edges(
        "reflect",
        _outer_transition,
        {
            "plan": "plan",
            "inquire": "inquire",
            END: END,
        },
    )
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)
    return builder.compile()


def resolve_graph(graph_name: str):
    normalized = str(graph_name).strip()
    if normalized == "builder":
        return build_builder_graph()
    if normalized == "executor":
        return build_executor_graph()
    if normalized == "with_report":
        return build_outer_graph()
    raise ValueError(f"Unknown graph {graph_name!r}; expected one of {sorted(VALID_GRAPHS)}.")


__all__ = [
    "VALID_GRAPHS",
    "build_builder_graph",
    "build_executor_graph",
    "build_outer_graph",
    "resolve_graph",
]
