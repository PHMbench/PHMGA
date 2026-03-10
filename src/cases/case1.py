from __future__ import annotations
import uuid
import yaml
from dotenv import load_dotenv
import os
from pathlib import Path
import time

# Load environment variables (best-effort; do not override existing env).
try:  # pragma: no cover
    load_dotenv(dotenv_path=str(Path.cwd() / ".env"), override=False)
except Exception:
    pass

# Disable LangSmith for cleaner logs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

from src.phm_outer_graph import build_builder_graph, build_executor_graph
from src.utils import initialize_state, initialize_state_vibench, save_state, load_state, generate_final_report
# from src.utils.visualization import visualize_dag_feature_evolution_umap
from src.agents.reflect_agent import get_dag_depth
from src.utils.preflight import build_preflight_report, write_preflight_report
from src.utils.logging_setup import (
    clear_current_logger,
    init_run_logger,
    log_event,
    set_current_logger,
    timed,
)

_MODEL_PROFILE_MAP = {
    "tspn_basic": "config/model_tspn_basic.yaml",
    "tspn_wf_heavy": "config/model_tspn_basic.yaml",
}
_ABLATION_MODES = {"full", "no_reflect", "no_prior"}
_LLM_PROVIDERS = {"gemini", "openai", "openai_compatible", "deepseek", "glm"}
_STATE_SAVE_MODES = {"auto", "full", "minimal"}


def _apply_state_update(state, update: dict) -> None:
    """Apply a node update dict onto a PHMState instance (ignore unknown keys).

    Some graph nodes return auxiliary keys (e.g., `new_nodes`, `n_nodes`) that are
    not part of the persisted PHMState schema; these should not crash the runner.
    """
    if not isinstance(update, dict):
        return
    fields = getattr(state.__class__, "model_fields", {})
    for key, value in update.items():
        if key in fields:
            setattr(state, key, value)


def _parse_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_source_mode(config: dict) -> str:
    data_cfg = dict(config.get("data") or {})
    source_mode = str(data_cfg.get("source_mode") or "").strip().lower()
    if source_mode in {"fixed_ids", "vibench"}:
        return source_mode
    backend = str(data_cfg.get("backend") or "").strip().lower()
    if backend == "vibench":
        return "vibench"
    return "fixed_ids"


def _resolve_state_save_mode(config: dict) -> str:
    data_cfg = dict(config.get("data") or {})
    mode = str(data_cfg.get("state_save_mode") or "auto").strip().lower() or "auto"
    if mode not in _STATE_SAVE_MODES:
        return "auto"
    return mode


def _missing_root_ref_channels(state) -> list[str]:
    missing: list[str] = []
    channels = list(getattr(state.dag_state, "channels", []) or [])
    for ch in channels:
        node = state.dag_state.nodes.get(ch)
        ref = (getattr(node, "results", {}) or {}).get("ref") if node is not None else None
        if not isinstance(ref, dict) or not ref:
            missing.append(str(ch))
    return missing


def _resolve_model_options(config: dict) -> dict:
    model_cfg = dict(config.get("model") or {})
    env_profile = os.getenv("PHM_MODEL_PROFILE", "").strip()
    profile = env_profile or str(model_cfg.get("profile") or "").strip()

    resolved_path = (
        config.get("model_config_path")
        or model_cfg.get("config_path")
        or (_MODEL_PROFILE_MAP.get(profile) if profile else None)
    )
    if not resolved_path:
        resolved_path = "config/model_tspn_basic.yaml"

    return {
        "profile": profile or "tspn_basic",
        "model_config_path": str(resolved_path),
        "autofit_dims": _parse_bool(model_cfg.get("autofit_dims"), default=True),
        "autofit_num_classes": _parse_bool(model_cfg.get("autofit_num_classes"), default=True),
    }


def _apply_runtime_overrides(config: dict) -> dict:
    out = dict(config)
    data_cfg = dict(out.get("data") or {})
    dataset_override = os.getenv("PHM_DATASET_NAME", "").strip()
    if dataset_override:
        data_cfg["dataset_name"] = dataset_override
    if data_cfg:
        out["data"] = data_cfg
    ablation_override = os.getenv("PHM_ABLATION_MODE", "").strip().lower()
    if ablation_override:
        ab_cfg = dict(out.get("ablation") or {})
        ab_cfg["mode"] = ablation_override
        out["ablation"] = ab_cfg
    return out


def _resolve_ablation_mode(config: dict) -> str:
    ab_cfg = dict(config.get("ablation") or {})
    mode = str(ab_cfg.get("mode") or "full").strip().lower()
    if mode not in _ABLATION_MODES:
        mode = "full"
    return mode


def _bind_llm_from_case(config: dict) -> str:
    llm_cfg = dict(config.get("llm") or {})
    if not llm_cfg:
        return "env"

    provider = str(llm_cfg.get("provider") or "").strip().lower()
    if provider not in _LLM_PROVIDERS:
        raise ValueError(
            f"Invalid llm.provider={provider!r}. "
            f"Expected one of: {', '.join(sorted(_LLM_PROVIDERS))}."
        )
    query_model = str(llm_cfg.get("query_generator_model") or "").strip()
    if not query_model:
        raise ValueError("llm.query_generator_model is required when llm block is provided.")

    phm_model = str(llm_cfg.get("phm_model") or query_model).strip()
    reflection_model = str(llm_cfg.get("reflection_model") or query_model).strip()
    answer_model = str(llm_cfg.get("answer_model") or query_model).strip()

    os.environ["LLM_PROVIDER"] = provider
    os.environ["QUERY_GENERATOR_MODEL"] = query_model
    os.environ["PHM_MODEL"] = phm_model
    os.environ["REFLECTION_MODEL"] = reflection_model
    os.environ["ANSWER_MODEL"] = answer_model

    config["llm"] = {
        "provider": provider,
        "query_generator_model": query_model,
        "phm_model": phm_model,
        "reflection_model": reflection_model,
        "answer_model": answer_model,
    }
    return "case_yaml"


def run_case(config_path: str):
    """
    Runs a full PHM analysis case based on a given configuration file.
    """
    # 1. Load configuration from YAML file
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    config = _apply_runtime_overrides(config)
    llm_source = _bind_llm_from_case(config)

    case_name = str(config.get("name") or "case")
    save_dir = str(config.get("save_dir") or os.environ.get("PHM_SAVE_DIR") or (Path.cwd() / "save"))
    run_id = f"run-{int(time.time())}"
    run_logger = init_run_logger(case_name=case_name, save_dir=save_dir, run_id=run_id)
    set_current_logger(run_logger)
    log_event(
        run_logger,
        level="INFO",
        event="case.load_config",
        phase="init",
        message="Loaded case configuration.",
        payload={"config_path": config_path, "run_id": run_id, "llm_source": llm_source},
    )

    preflight = build_preflight_report(config)
    preflight_path = run_logger.log_dir / "preflight_report.json"
    write_preflight_report(preflight, preflight_path)
    log_event(
        run_logger,
        level="INFO",
        event="case.preflight",
        phase="init",
        message="Preflight completed.",
        payload={"ok": preflight.get("ok"), "errors": preflight.get("errors"), "warnings": preflight.get("warnings")},
    )
    preflight_cfg = dict(config.get("preflight") or {})
    if not preflight.get("ok") and _parse_bool(preflight_cfg.get("strict"), default=True):
        raise ValueError(f"Preflight failed. See {preflight_path}")

    state_save_path = config["state_save_path"]
    builder_cfg = config.get('builder', {})
    min_depth = builder_cfg.get('min_depth', 0)
    max_depth = builder_cfg.get('max_depth', float('inf'))
    max_iterations = int(builder_cfg.get("max_iterations", 50))
    ablation_mode = _resolve_ablation_mode(config)
    if ablation_mode == "no_reflect":
        max_iterations = min(max_iterations, 1)
    source_mode = _resolve_source_mode(config)
    state_save_mode = _resolve_state_save_mode(config)
    model_opts = _resolve_model_options(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["source_mode"] = source_mode
    data_cfg["state_save_mode"] = state_save_mode
    data_cfg["model_profile"] = model_opts["profile"]
    data_cfg["autofit_dims"] = model_opts["autofit_dims"]
    data_cfg["autofit_num_classes"] = model_opts["autofit_num_classes"]
    data_cfg["model_config_path"] = model_opts["model_config_path"]
    data_cfg["ablation_mode"] = ablation_mode
    data_cfg["disable_prior_init"] = (ablation_mode == "no_prior")
    log_event(
        run_logger,
        level="INFO",
        event="case.ablation",
        phase="init",
        message="Resolved ablation mode.",
        payload={
            "ablation_mode": ablation_mode,
            "max_iterations": max_iterations,
            "source_mode": source_mode,
            "state_save_mode": state_save_mode,
        },
    )
    if source_mode == "fixed_ids":
        data_cfg.setdefault("metadata_path", config.get("metadata_path"))
        data_cfg.setdefault("h5_path", config.get("h5_path"))
        data_cfg.setdefault("ref_ids", config.get("ref_ids"))
        data_cfg.setdefault("test_ids", config.get("test_ids"))

    # --- Check for existing state ---
    if os.path.exists(state_save_path):
        log_event(
            run_logger,
            level="INFO",
            event="case.state_found",
            phase="init",
            message="Existing state file found. Skip builder.",
            payload={"state_save_path": state_save_path},
        )
        built_state = load_state(state_save_path)
        if built_state is None:
            log_event(
                run_logger,
                level="ERROR",
                event="case.state_load_failed",
                phase="init",
                message="Failed to load state from file.",
                payload={"state_save_path": state_save_path},
            )
            clear_current_logger()
            return
        built_state.train_backend = str(config.get("train_backend", built_state.train_backend or "tspn"))
        built_state.model_config_path = model_opts["model_config_path"]
        built_state.data_cfg = data_cfg
        built_state.save_dir = config.get("save_dir")
        built_state.max_builder_iterations = max_iterations
        if source_mode == "fixed_ids":
            missing = _missing_root_ref_channels(built_state)
            if missing:
                msg = (
                    "Loaded built_state.pkl is missing root InputData.results['ref'] "
                    f"for channels={missing}. fixed_ids requires full state arrays. "
                    "Use data.state_save_mode=full and rebuild (or remove current built_state.pkl)."
                )
                log_event(
                    run_logger,
                    level="ERROR",
                    event="case.state_incompatible",
                    phase="init",
                    message=msg,
                    payload={"state_save_path": state_save_path, "missing_channels": missing},
                )
                clear_current_logger()
                raise ValueError(msg)
    else:
        log_event(
            run_logger,
            level="INFO",
            event="case.state_missing",
            phase="init",
            message="No existing state file found. Start builder workflow.",
            payload={"state_save_path": state_save_path},
        )
        # --- Part 0: Initialization ---
        log_event(run_logger, level="INFO", event="case.part0.start", phase="init", message="Initializing PHMState.")
        with timed(run_logger, event="case.part0.init_state", phase="init"):
            if source_mode == "vibench":
                data_cfg["backend"] = "vibench"
                initial_phm_state = initialize_state_vibench(
                    user_instruction=config["user_instruction"],
                    case_name=config["name"],
                    data_cfg=data_cfg,
                    allow_test_labels_for_reporting=bool(config.get("allow_test_labels_for_reporting", False)),
                    train_backend=str(config.get("train_backend", "tspn")),
                    model_config_path=model_opts["model_config_path"],
                    save_dir=config.get("save_dir"),
                )
            else:
                metadata_path = str(config.get("metadata_path") or data_cfg.get("metadata_path") or "")
                h5_path = str(config.get("h5_path") or data_cfg.get("h5_path") or "")
                ref_ids = list(config.get("ref_ids") or data_cfg.get("ref_ids") or [])
                test_ids = list(config.get("test_ids") or data_cfg.get("test_ids") or [])
                initial_phm_state = initialize_state(
                    user_instruction=config["user_instruction"],
                    metadata_path=metadata_path,
                    h5_path=h5_path,
                    ref_ids=ref_ids,
                    test_ids=test_ids,
                    case_name=config["name"],
                    allow_test_labels_for_reporting=bool(config.get("allow_test_labels_for_reporting", False)),
                    train_backend=str(config.get("train_backend", "shallow")),
                    model_config_path=model_opts["model_config_path"],
                    save_dir=config.get("save_dir"),
                    data_cfg=data_cfg,
                )
            initial_phm_state.data_cfg = data_cfg
            initial_phm_state.max_builder_iterations = max_iterations

        # --- Part 1: Run DAG Builder Workflow ---
        log_event(run_logger, level="INFO", event="case.part1.start", phase="builder", message="Starting DAG builder workflow.")
        builder_app = build_builder_graph()

        built_state = initial_phm_state.model_copy(deep=True)
        iteration = 0

        while True:
            if int(getattr(built_state, "iteration_count", 0) or 0) >= max_iterations:
                log_event(
                    run_logger,
                    level="WARNING",
                    event="builder.stop.max_iterations",
                    phase="builder",
                    message="Reached builder max iterations safety limit.",
                    payload={
                        "iteration_count": int(getattr(built_state, "iteration_count", 0) or 0),
                        "max_iterations": max_iterations,
                    },
                )
                built_state.needs_revision = False
                break
            iteration += 1
            if iteration > max_iterations:
                log_event(
                    run_logger,
                    level="WARNING",
                    event="builder.stop.max_iterations",
                    phase="builder",
                    message="Reached builder max iterations safety limit.",
                    payload={"iteration": iteration - 1, "max_iterations": max_iterations},
                )
                built_state.needs_revision = False
                break
            log_event(
                run_logger,
                level="INFO",
                event="builder.iteration.start",
                phase="builder",
                message="Builder iteration started.",
                payload={"iteration": iteration},
            )
            thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            with timed(run_logger, event="builder.iteration", phase="builder", payload={"iteration": iteration}):
                for event in builder_app.stream(built_state, config=thread_config):
                    for node_name, state_update in event.items():
                        log_event(
                            run_logger,
                            level="INFO",
                            event="builder.node.executed",
                            phase="builder",
                            node=node_name,
                            message="Builder node executed.",
                            payload={"has_update": state_update is not None},
                        )
                        if state_update is not None:
                            _apply_state_update(built_state, state_update)

            depth = get_dag_depth(built_state.dag_state)
            log_event(
                run_logger,
                level="INFO",
                event="builder.depth",
                phase="builder",
                message="Current DAG depth.",
                payload={"depth": depth, "min_depth": min_depth, "max_depth": max_depth},
            )

            if depth >= max_depth:
                log_event(
                    run_logger,
                    level="INFO",
                    event="builder.stop.max_depth",
                    phase="builder",
                    message="Reached max depth.",
                    payload={"depth": depth, "max_depth": max_depth},
                )
                break

            if depth < min_depth:
                log_event(
                    run_logger,
                    level="INFO",
                    event="builder.force_continue",
                    phase="builder",
                    message="Depth below min_depth; continue regardless of reflection.",
                    payload={"depth": depth, "min_depth": min_depth},
                )
                built_state.needs_revision = True

            if not built_state.needs_revision:
                log_event(
                    run_logger,
                    level="INFO",
                    event="builder.stop.reflect",
                    phase="builder",
                    message="Reflect agent indicated to stop.",
                )
                break

        log_event(run_logger, level="INFO", event="case.part1.finish", phase="builder", message="DAG builder workflow finished.")
        if not built_state:
            log_event(
                run_logger,
                level="ERROR",
                event="builder.empty_state",
                phase="builder",
                message="Builder workflow failed to produce final state.",
            )
            clear_current_logger()
            return

        log_event(
            run_logger,
            level="INFO",
            event="builder.summary",
            phase="builder",
            message="Built DAG summary.",
            payload={
                "leaves": list(built_state.dag_state.leaves),
                "n_nodes": len(built_state.dag_state.nodes),
                "n_errors": len(built_state.dag_state.error_log),
            },
        )

        # --- Save the built state ---
        with timed(
            run_logger,
            event="case.save_state",
            phase="builder",
            payload={
                "state_save_path": state_save_path,
                "save_mode": state_save_mode,
                "source_mode": source_mode,
            },
        ):
            save_state(
                built_state,
                state_save_path,
                save_mode=state_save_mode,
                source_mode=source_mode,
            )

    # At this point, `built_state` is guaranteed to be a valid state object,
    # either loaded from file or newly created.

    # --- Part 2: Run DAG Executor Workflow (optional) ---
    if bool(config.get("run_executor", False)):
        log_event(run_logger, level="INFO", event="case.part2.start", phase="executor", message="Starting DAG executor workflow.")
        executor_app = build_executor_graph()
        thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Use a new thread for the executor

        final_state = built_state.model_copy(deep=True)
        with timed(run_logger, event="case.part2.executor", phase="executor"):
            for event in executor_app.stream(built_state, config=thread_config):
                for node_name, state_update in event.items():
                    log_event(
                        run_logger,
                        level="INFO",
                        event="executor.node.executed",
                        phase="executor",
                        node=node_name,
                        message="Executor node executed.",
                        payload={"has_update": state_update is not None},
                    )
                    if state_update is not None:
                        _apply_state_update(final_state, state_update)

        # --- Part 3: Generate Final Report ---
        with timed(run_logger, event="case.part3.report", phase="report", payload={"report_path": config['report_path']}):
            generate_final_report(final_state, config['report_path'])
    else:
        log_event(run_logger, level="INFO", event="case.executor.skipped", phase="executor", message="Executor disabled by config.")

    log_event(
        run_logger,
        level="INFO",
        event="case.finish",
        phase="report",
        message="Case run finished.",
        payload={"state_save_path": state_save_path, "report_path": config.get("report_path")},
    )
    clear_current_logger()

    # # --- Part 3: Visualize Feature Evolution ---
    # root_node = next(iter(final_state.dag_state.nodes.values()))
    # labels = list(root_node.meta.get("labels", {}).values())
    # visualize_dag_feature_evolution_umap(final_state.dag_state, final_state, labels)

    # # --- Part 4: Generate Final Report ---
    # generate_final_report(final_state, config['report_path'])

if __name__ == "__main__":
    # This allows running the case directly for testing
    # run_case("config/case1.yaml")
    # run_case("config/case_exp2.yaml")
    # run_case("config/case_exp2.5.yaml")
    run_case("config/case_exp_ottawa.yaml")
