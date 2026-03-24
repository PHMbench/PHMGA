"""Public API for the graph-first PHMGA package."""

try:
    from .graph import graph
except Exception:  # pragma: no cover - optional dependency may be missing
    graph = None

try:
    from .phm_outer_graph import (
        build_builder_graph,
        build_executor_graph,
        build_outer_graph,
        resolve_graph,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    build_builder_graph = None
    build_executor_graph = None
    build_outer_graph = None
    resolve_graph = None

try:
    from .manual_workflow import (
        export_dag_artifacts,
        generate_report_from_state,
        load_node_dataset,
        load_node_datasets,
        prepare_datasets_from_state,
        run_manual_postprocess,
        save_node_datasets,
        train_ml_from_datasets,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    export_dag_artifacts = None
    generate_report_from_state = None
    load_node_dataset = None
    load_node_datasets = None
    prepare_datasets_from_state = None
    run_manual_postprocess = None
    save_node_datasets = None
    train_ml_from_datasets = None

try:
    from .evaluation import (
        build_input_split_results,
        build_root_split_maps,
        build_split_labels,
        compute_cross_dag_late_fusion,
        evaluate_full_dag_leaves,
        extract_leaf_datasets,
        load_run_prediction_payload,
        replay_state_on_split_maps,
        replay_state_on_split_results,
        summarize_branch,
        summarize_dag_state,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    build_input_split_results = None
    build_root_split_maps = None
    build_split_labels = None
    compute_cross_dag_late_fusion = None
    evaluate_full_dag_leaves = None
    extract_leaf_datasets = None
    load_run_prediction_payload = None
    replay_state_on_split_maps = None
    replay_state_on_split_results = None
    summarize_branch = None
    summarize_dag_state = None

try:
    from .simulated_rm101 import (
        SIMULATED_MODEL_TAGS,
        build_simulated_state,
        get_simulated_run_info,
        validate_complexity_ladder,
        validate_simulated_state,
    )
except Exception:  # pragma: no cover - optional dependency may be missing
    SIMULATED_MODEL_TAGS = None
    build_simulated_state = None
    get_simulated_run_info = None
    validate_complexity_ladder = None
    validate_simulated_state = None

from .config import load_case_config, resolve_case_path
from .model import get_llm

__all__ = [
    "build_builder_graph",
    "build_executor_graph",
    "build_outer_graph",
    "export_dag_artifacts",
    "build_input_split_results",
    "build_root_split_maps",
    "build_split_labels",
    "build_simulated_state",
    "compute_cross_dag_late_fusion",
    "evaluate_full_dag_leaves",
    "extract_leaf_datasets",
    "generate_report_from_state",
    "graph",
    "get_llm",
    "get_simulated_run_info",
    "load_node_dataset",
    "load_node_datasets",
    "load_run_prediction_payload",
    "prepare_datasets_from_state",
    "replay_state_on_split_maps",
    "replay_state_on_split_results",
    "run_manual_postprocess",
    "save_node_datasets",
    "SIMULATED_MODEL_TAGS",
    "summarize_branch",
    "summarize_dag_state",
    "train_ml_from_datasets",
    "validate_complexity_ladder",
    "validate_simulated_state",
    "load_case_config",
    "resolve_case_path",
    "resolve_graph",
]
