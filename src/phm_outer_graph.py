from __future__ import annotations

import traceback
from typing import Any, Dict

try:  # pragma: no cover
    from langgraph.graph import StateGraph, END, START  # type: ignore
    _LANGGRAPH_OK = True
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = "__END__"  # type: ignore
    START = "__START__"  # type: ignore
    _LANGGRAPH_OK = False

from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.dag_init_agent import dag_init_agent
from .agents.deep_model_train_agent import deep_model_train_agent
from .agents.execute_agent import execute_agent
from .agents.inquirer_agent import inquirer_agent
from .agents.plan_agent import plan_agent
from .agents.reflect_agent import reflect_agent_node
from .agents.report_agent import report_agent_node
from .agents.tspn_bootstrap_agent import tspn_bootstrap_agent
from .states.phm_states import PHMState
from .utils.logging_setup import get_current_logger, log_event, timed


class _FallbackGraph:
    def __init__(self, steps):
        self._steps = list(steps)

    def stream(self, state: PHMState, config: Any | None = None):
        # Mimic langgraph streaming API: yield {node_name: update_dict}
        for name, fn in self._steps:
            update = _run_node(name, fn, state)
            if isinstance(update, dict):
                # Apply updates in-place so downstream nodes can observe them.
                fields = getattr(state.__class__, "model_fields", {})
                for k, v in update.items():
                    if k in fields:
                        setattr(state, k, v)
            yield {name: update}


def _summarize_update(update: Any) -> Dict[str, Any]:
    if not isinstance(update, dict):
        return {"update_type": type(update).__name__}
    summary: Dict[str, Any] = {"keys": sorted(update.keys())}
    lens: Dict[str, int] = {}
    for key, value in update.items():
        if isinstance(value, (list, tuple, dict, set)):
            try:
                lens[key] = len(value)
            except Exception:
                continue
    if lens:
        summary["lengths"] = lens
    return summary


def _run_node(name: str, fn: Any, state: PHMState) -> Dict[str, Any]:
    logger = get_current_logger()
    with timed(logger, event="node", phase="graph", node=name, message="Node execution"):
        try:
            update = fn(state)
            log_event(
                logger,
                level="INFO",
                event="state_update",
                phase="graph",
                node=name,
                message="Node returned state update.",
                payload=_summarize_update(update),
            )
            return update
        except Exception as exc:
            log_event(
                logger,
                level="ERROR",
                event="node.exception",
                phase="graph",
                node=name,
                message=str(exc),
                payload={"exc_type": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            raise


def build_builder_graph() -> Any:
    """
    Constructs the graph responsible for iteratively building the computational DAG.
    This graph uses a plan-execute-reflect loop to generate a valid DAG.
    """
    if not _LANGGRAPH_OK:  # pragma: no cover
        # Fallback for environments with incompatible langgraph/langchain_core versions.
        # The caller (case runner) controls the outer loop; we run plan->execute->reflect once per `.stream()`.
        return _FallbackGraph(
            [
                ("plan", plan_agent),
                ("execute", execute_agent),
                ("reflect", lambda state: reflect_agent_node(state, stage="POST_EXECUTE")),
            ]
        )

    builder = StateGraph(PHMState)  # type: ignore[misc]

    builder.add_node("plan", lambda state: _run_node("plan", plan_agent, state))
    builder.add_node("execute", lambda state: _run_node("execute", execute_agent, state))
    builder.add_node(
        "reflect", lambda state: _run_node("reflect", lambda s: reflect_agent_node(s, stage="POST_EXECUTE"), state)
    )

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


def build_executor_graph() -> Any:
    """
    Constructs the graph that executes a finalized computational DAG.
    This graph performs similarity analysis, dataset preparation, model training, and reporting.
    """
    def _train_models(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        ml_results: Dict[str, Any] = dict(getattr(state, "ml_results", {}) or {})

        if backend in {"shallow", "both"}:
            try:
                from .agents.shallow_ml_agent import shallow_ml_agent  # local import (optional dependency: joblib/sklearn)

                shallow = shallow_ml_agent(datasets=state.datasets)
                # Backward-compatible top-level fields for report prompt.
                ml_results.update(shallow)
                ml_results["shallow"] = shallow
            except Exception as e:
                ml_results["shallow"] = {"error": f"shallow_ml_agent unavailable: {type(e).__name__}: {e}"}

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

    def _init_dag_for_tspn(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        if backend not in {"tspn", "both"}:
            return {}
        # If builder already produced processed nodes, no need to re-init.
        if any(getattr(n, "method", None) for n in (state.dag_state.nodes or {}).values()):
            return {}
        return dag_init_agent(state)

    def _bootstrap_tspn_config(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        if backend not in {"tspn", "both"}:
            return {}
        data_cfg = dict(getattr(state, "data_cfg", {}) or {})
        if str(data_cfg.get("backend") or "").strip().lower() == "vibench":
            # vibench path uses DAG2ConfigAdapter inside deep_model_train_agent; no YAML bootstrap here.
            return {}
        if getattr(state, "model_config_path", None):
            return {}
        # Deterministic bootstrap from the built DAG to reduce hand-designed priors.
        return tspn_bootstrap_agent(state)

    def _executor_path(state: PHMState) -> str:
        logger = get_current_logger()
        backend = (getattr(state, "train_backend", None) or "shallow").strip().lower()
        path = "tspn_fast_path" if backend == "tspn" else "full_path"
        log_event(
            logger,
            level="INFO",
            event="executor.path",
            phase="executor",
            node="route",
            message="Selected executor path.",
            payload={"path": path, "train_backend": backend},
        )
        return path

    if not _LANGGRAPH_OK:  # pragma: no cover
        class _ExecutorFallbackGraph:
            def stream(self, state: PHMState, config: Any | None = None):
                path = _executor_path(state)
                if path == "tspn_fast_path":
                    steps = [
                        ("init_dag", _init_dag_for_tspn),
                        ("bootstrap", _bootstrap_tspn_config),
                        ("train", _train_models),
                        ("report", report_agent_node),
                    ]
                else:
                    steps = [
                        ("inquire", lambda s: inquirer_agent(s, metrics=["cosine", "euclidean"])),
                        ("prepare", dataset_preparer_agent),
                        ("init_dag", _init_dag_for_tspn),
                        ("bootstrap", _bootstrap_tspn_config),
                        ("train", _train_models),
                        ("report", report_agent_node),
                    ]
                for name, fn in steps:
                    update = _run_node(name, fn, state)
                    if isinstance(update, dict):
                        fields = getattr(state.__class__, "model_fields", {})
                        for k, v in update.items():
                            if k in fields:
                                setattr(state, k, v)
                    yield {name: update}

        return _ExecutorFallbackGraph()

    builder = StateGraph(PHMState)  # type: ignore[misc]

    # Define the nodes for the execution pipeline
    builder.add_node("route", lambda state: _run_node("route", lambda s: {}, state))
    builder.add_node(
        "inquire", lambda state: _run_node("inquire", lambda s: inquirer_agent(s, metrics=["cosine", "euclidean"]), state)
    )
    builder.add_node("prepare", lambda state: _run_node("prepare", dataset_preparer_agent, state))
    builder.add_node("init_dag", lambda state: _run_node("init_dag", _init_dag_for_tspn, state))
    builder.add_node("bootstrap", lambda state: _run_node("bootstrap", _bootstrap_tspn_config, state))
    builder.add_node("train", lambda state: _run_node("train", _train_models, state))
    builder.add_node("report", lambda state: _run_node("report", report_agent_node, state))

    # Define the execution flow with runtime routing.
    builder.set_entry_point("route")
    builder.add_conditional_edges(
        "route",
        _executor_path,
        {
            "tspn_fast_path": "init_dag",
            "full_path": "inquire",
        },
    )
    builder.add_edge("inquire", "prepare")
    builder.add_edge("prepare", "init_dag")
    builder.add_edge("init_dag", "bootstrap")
    builder.add_edge("bootstrap", "train")
    builder.add_edge("train", "report")
    builder.add_edge("report", END)

    return builder.compile()


# Backward-compatible alias for older tests/examples.
def build_outer_graph() -> Any:  # pragma: no cover
    """
    Legacy entry-point kept for compatibility.

    The codebase now exposes two decoupled graphs:
    - build_builder_graph(): iteratively constructs a DAG
    - build_executor_graph(): executes a finalized DAG

    For legacy callers, we return the builder graph.
    """
    return build_builder_graph()
