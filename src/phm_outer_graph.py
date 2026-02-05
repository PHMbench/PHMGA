from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END, START

from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.deep_model_train_agent import deep_model_train_agent
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
        lambda state: "plan" if state.needs_revision else END,
        {
            "plan": "plan",
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

    def _train_models(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        ml_results: Dict[str, Any] = dict(getattr(state, "ml_results", {}) or {})

        if backend in {"shallow", "both"}:
            shallow = shallow_ml_agent(datasets=state.datasets)
            # Backward-compatible top-level fields for report prompt.
            ml_results.update(shallow)
            ml_results["shallow"] = shallow

        if backend in {"tspn", "both"}:
            tmp_state = state.model_copy(deep=False)
            tmp_state.ml_results = ml_results
            out = deep_model_train_agent(tmp_state)
            if "ml_results" in out:
                ml_results = out["ml_results"]
            updates = {"ml_results": ml_results}
            if "run_dir" in out:
                updates["run_dir"] = out["run_dir"]
            return updates

        return {"ml_results": ml_results}

    # Define the nodes for the execution pipeline
    builder.add_node("inquire", lambda state: inquirer_agent(state, metrics=["cosine", "euclidean"]))
    builder.add_node("prepare", dataset_preparer_agent)
    builder.add_node("train", _train_models)
    builder.add_node("report", report_agent_node)

    # Define the linear flow of the execution graph
    builder.set_entry_point("inquire")
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)

    return builder.compile()


# Backward-compatible alias for older tests/examples.
def build_outer_graph() -> StateGraph:  # pragma: no cover
    """
    Legacy entry-point kept for compatibility.

    The codebase now exposes two decoupled graphs:
    - build_builder_graph(): iteratively constructs a DAG
    - build_executor_graph(): executes a finalized DAG

    For legacy callers, we return the builder graph.
    """
    return build_builder_graph()
