from __future__ import annotations
import uuid
import yaml
from dotenv import load_dotenv
import os
from pathlib import Path

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


def run_case(config_path: str):
    """
    Runs a full PHM analysis case based on a given configuration file.
    """
    # 1. Load configuration from YAML file
    print(f"--- Loading configuration from {config_path} ---")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    state_save_path = config['state_save_path']
    builder_cfg = config.get('builder', {})
    min_depth = builder_cfg.get('min_depth', 0)
    max_depth = builder_cfg.get('max_depth', float('inf'))

    # --- Check for existing state ---
    if os.path.exists(state_save_path):
        print(f"\n--- Found existing state file at {state_save_path}. Skipping builder workflow. ---")
        built_state = load_state(state_save_path)
        if built_state is None:
            print("Failed to load state. Exiting.")
            return
    else:
        print(f"\n--- No existing state file found. Starting builder workflow. ---")
        # --- Part 0: Initialization ---
        print("\n--- [Part 0] Initializing State ---")
        data_cfg = config.get("data") or {}
        if str((data_cfg.get("backend") or "")).strip().lower() == "vibench":
            initial_phm_state = initialize_state_vibench(
                user_instruction=config["user_instruction"],
                case_name=config["name"],
                data_cfg=data_cfg,
                allow_test_labels_for_reporting=bool(config.get("allow_test_labels_for_reporting", False)),
                train_backend=str(config.get("train_backend", "tspn")),
                model_config_path=config.get("model_config_path"),
                save_dir=config.get("save_dir"),
            )
        else:
            initial_phm_state = initialize_state(
                user_instruction=config["user_instruction"],
                metadata_path=config["metadata_path"],
                h5_path=config["h5_path"],
                ref_ids=config["ref_ids"],
                test_ids=config["test_ids"],
                case_name=config["name"],
                allow_test_labels_for_reporting=bool(config.get("allow_test_labels_for_reporting", False)),
                train_backend=str(config.get("train_backend", "shallow")),
                model_config_path=config.get("model_config_path"),
                save_dir=config.get("save_dir"),
            )

        # --- Part 1: Run DAG Builder Workflow ---
        print("\n--- [Part 1] Starting DAG Builder Workflow ---")
        builder_app = build_builder_graph()

        built_state = initial_phm_state.model_copy(deep=True)
        iteration = 0

        while True:
            iteration += 1
            print(f"\n--- Builder Iteration {iteration} ---")
            thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            for event in builder_app.stream(built_state, config=thread_config):
                for node_name, state_update in event.items():
                    print(f"--- Builder Node Executed: {node_name} ---")
                    if state_update is not None:
                        _apply_state_update(built_state, state_update)

            depth = get_dag_depth(built_state.dag_state)
            print(f"Current DAG depth: {depth}")

            if depth >= max_depth:
                print(f"Reached max depth {max_depth}. Stopping builder.")
                break

            if depth < min_depth:
                print(f"Depth {depth} below min_depth {min_depth}. Continuing regardless of reflection.")
                built_state.needs_revision = True

            if not built_state.needs_revision:
                print("Reflect agent indicated to stop.")
                break

        print("\n--- [Part 1] DAG Builder Workflow Finished ---")
        if not built_state:
            print("Builder workflow failed to produce a final state.")
            return
            
        print(f"Final leaves of the built DAG: {built_state.dag_state.leaves}")
        print(f"Total nodes in DAG: {len(built_state.dag_state.nodes)}")
        print(f"Errors during build: {built_state.dag_state.error_log}")

        # --- Save the built state ---
        save_state(built_state, state_save_path)

    # At this point, `built_state` is guaranteed to be a valid state object,
    # either loaded from file or newly created.

    # --- Part 2: Run DAG Executor Workflow (optional) ---
    if bool(config.get("run_executor", False)):
        print("\n--- [Part 2] Starting DAG Executor Workflow ---")
        executor_app = build_executor_graph()
        thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # Use a new thread for the executor

        final_state = built_state.model_copy(deep=True)
        for event in executor_app.stream(built_state, config=thread_config):
            for node_name, state_update in event.items():
                print(f"--- Executor Node Executed: {node_name} ---")
                if state_update is not None:
                    _apply_state_update(final_state, state_update)

        # --- Part 3: Generate Final Report ---
        generate_final_report(final_state, config['report_path'])

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
