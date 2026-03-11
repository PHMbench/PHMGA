from __future__ import annotations

from typing import Any, Dict

from src.agents.dag_init_agent import dag_init_agent
from src.agents.dataset_preparer_agent import dataset_preparer_agent
from src.agents.deep_model_train_agent import deep_model_train_agent
from src.agents.inquirer_agent import inquirer_agent
from src.agents.report_agent import report_agent_node
from src.agents.tspn_bootstrap_agent import tspn_bootstrap_agent
from src.states.phm_states import PHMState
from src.utils.logging_setup import get_current_logger, log_event

from .common import END, LANGGRAPH_OK, FallbackGraph, StateGraph, run_node


def build_executor_graph() -> Any:
    def _train_models(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        allowed_backends = {"shallow", "tspn", "both"}
        if backend not in allowed_backends:
            raise ValueError(
                f"Invalid train_backend '{backend}'. Expected one of: {sorted(allowed_backends)}"
            )
        ml_results: Dict[str, Any] = dict(getattr(state, "ml_results", {}) or {})

        if backend in {"shallow", "both"}:
            try:
                from src.agents.shallow_ml_agent import shallow_ml_agent

                shallow = shallow_ml_agent(datasets=state.datasets)
                ml_results.update(shallow)
                ml_results["shallow"] = shallow
            except Exception as exc:
                ml_results["shallow"] = {"error": f"shallow_ml_agent unavailable: {type(exc).__name__}: {exc}"}

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
        if any(getattr(node, "method", None) for node in (state.dag_state.nodes or {}).values()):
            return {}
        return dag_init_agent(state)

    def _bootstrap_tspn_config(state: PHMState) -> dict:
        backend = (getattr(state, "train_backend", None) or "shallow").lower()
        if backend not in {"tspn", "both"}:
            return {}
        data_cfg = dict(getattr(state, "data_cfg", {}) or {})
        if str(data_cfg.get("backend") or "").strip().lower() == "vibench":
            return {}
        if getattr(state, "model_config_path", None):
            return {}
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

    if not LANGGRAPH_OK:  # pragma: no cover
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
                return FallbackGraph(steps).stream(state, config=config)

        return _ExecutorFallbackGraph()

    builder = StateGraph(PHMState)  # type: ignore[misc]
    builder.add_node("route", lambda state: run_node("route", lambda s: {}, state))
    builder.add_node(
        "inquire",
        lambda state: run_node("inquire", lambda s: inquirer_agent(s, metrics=["cosine", "euclidean"]), state),
    )
    builder.add_node("prepare", lambda state: run_node("prepare", dataset_preparer_agent, state))
    builder.add_node("init_dag", lambda state: run_node("init_dag", _init_dag_for_tspn, state))
    builder.add_node("bootstrap", lambda state: run_node("bootstrap", _bootstrap_tspn_config, state))
    builder.add_node("train", lambda state: run_node("train", _train_models, state))
    builder.add_node("report", lambda state: run_node("report", report_agent_node, state))
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
