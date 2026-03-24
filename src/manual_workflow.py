from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .agents.dataset_preparer_agent import dataset_preparer_agent
from .agents.report_agent import report_agent_node
from .agents.shallow_ml_agent import shallow_ml_agent
from .dag_artifacts import export_state_artifacts
from .states.phm_states import PHMState


def export_dag_artifacts(
    state: PHMState,
    *,
    output_dir: str | Path,
    stem: str = "dag",
    max_nodes: int | None = None,
    save_png: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    """Export the current DAG as reusable artifacts without running LangGraph."""
    return export_state_artifacts(
        state,
        output_dir=output_dir,
        stem=stem,
        max_nodes=max_nodes,
        save_png=save_png,
        save_json=save_json,
    )


def prepare_datasets_from_state(
    state: PHMState,
    *,
    stage: str = "processed",
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Build datasets directly from a saved/built PHMState."""
    result = dataset_preparer_agent(state, config={"stage": stage})
    datasets = result.get("datasets", {})
    saved_paths = save_node_datasets(datasets, output_dir) if output_dir else []
    return {
        "datasets": datasets,
        "n_nodes": result.get("n_nodes", len(datasets)),
        "saved_paths": saved_paths,
    }


def save_node_datasets(datasets: Dict[str, Dict[str, Any]], output_dir: str | Path) -> list[str]:
    """Persist per-node train/test datasets as ``.npz`` files."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    for node_id, data in datasets.items():
        path = target_dir / f"{node_id}_dataset.npz"
        np.savez(
            path,
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_test=data["X_test"],
            y_test=data["y_test"],
        )
        saved_paths.append(str(path))
    return saved_paths


def load_node_dataset(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load one per-node dataset exported by :func:`save_node_datasets`."""
    with np.load(npz_path) as data:
        return {
            "X_train": data["X_train"],
            "y_train": data["y_train"],
            "X_test": data["X_test"],
            "y_test": data["y_test"],
        }


def load_node_datasets(folder: str | Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all saved node datasets from a folder."""
    folder_path = Path(folder)
    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    for npz_path in sorted(folder_path.glob("*_dataset.npz")):
        node_id = npz_path.name.replace("_dataset.npz", "")
        datasets[node_id] = load_node_dataset(npz_path)
    return datasets


def train_ml_from_datasets(
    datasets: Dict[str, Dict[str, Any]],
    *,
    results_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train shallow ML models outside the graph and optionally persist the results."""
    results = shallow_ml_agent(datasets, **kwargs)
    if markdown_path is not None:
        markdown_target = Path(markdown_path)
        markdown_target.parent.mkdir(parents=True, exist_ok=True)
        markdown_target.write_text(str(results.get("metrics_markdown", "")), encoding="utf-8")
    if results_path is not None:
        results_target = Path(results_path)
        results_target.parent.mkdir(parents=True, exist_ok=True)
        with results_target.open("wb") as fh:
            pickle.dump(results, fh)
    return results


def generate_report_from_state(
    state: PHMState,
    *,
    ml_results: Dict[str, Any] | None = None,
    report_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Generate a final markdown report directly from state data."""
    working_state = state.model_copy(deep=True)
    if ml_results is not None:
        working_state.ml_results = ml_results
    out = report_agent_node(working_state)
    markdown = out["final_report"]
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding="utf-8")
    return {"final_report": markdown}


def run_manual_postprocess(
    state: PHMState,
    *,
    stage: str = "processed",
    dataset_output_dir: str | Path | None = None,
    ml_results_path: str | Path | None = None,
    metrics_markdown_path: str | Path | None = None,
    report_path: str | Path | None = None,
    **ml_kwargs: Any,
) -> Dict[str, Any]:
    """Run dataset preparation, shallow ML, and report generation without LangGraph orchestration."""
    dataset_result = prepare_datasets_from_state(state, stage=stage, output_dir=dataset_output_dir)
    ml_results = train_ml_from_datasets(
        dataset_result["datasets"],
        results_path=ml_results_path,
        markdown_path=metrics_markdown_path,
        **ml_kwargs,
    )
    report_result = generate_report_from_state(state, ml_results=ml_results, report_path=report_path)
    return {
        "datasets": dataset_result["datasets"],
        "n_nodes": dataset_result["n_nodes"],
        "saved_dataset_paths": dataset_result["saved_paths"],
        "ml_results": ml_results,
        "final_report": report_result["final_report"],
    }
