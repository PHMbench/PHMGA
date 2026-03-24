from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from src.agents.shallow_ml_agent import shallow_ml_agent
from src.data import (
    build_protocol_from_config,
    export_split_manifest,
    summarize_protocol,
    summarize_split_ids_by_label_domain,
    summarize_split_records,
)
from src.data.protocol import SignalRecord, materialize_split_signals
from src.states.phm_states import PHMState

from .replay import (
    build_input_split_results,
    build_split_labels,
    extract_leaf_datasets,
    replay_state_on_split_maps,
)


def summarize_dag_state(state: PHMState) -> Dict[str, Any]:
    graph = state.tracker().g
    depth = 0
    if graph.nodes:
        if graph.edges:
            depth = nx.dag_longest_path_length(graph) + 1
        else:
            depth = 1
    unique_ops = sorted(
        {
            str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", "") or "").strip()
            for node in state.dag_state.nodes.values()
            if str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", "") or "").strip()
        }
    )
    return {
        "depth": depth,
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "unique_ops": unique_ops,
        "leaf_count": len(state.dag_state.leaves),
    }


def evaluate_full_dag_leaves(
    state: PHMState,
    *,
    case_config: Optional[Dict[str, Any]] = None,
    split_records: Optional[Dict[str, List[SignalRecord]]] = None,
    cv_folds: int = 5,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    candidate_algorithms: Optional[List[str]] = None,
    parallel_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Run split-before-windowing, replay the full DAG, and score terminal leaves."""
    protocol_summary: Dict[str, Any] = {}
    split_manifest: Dict[str, Any] = {}
    split_summary_by_label_domain: List[Dict[str, Any]] = []
    if split_records is not None:
        root_split_maps = build_input_split_results(split_records, channel_ids=list(state.dag_state.channels))
        labels_by_split = build_split_labels(split_records)
        protocol_summary = summarize_split_records(split_records)
    elif case_config is not None:
        protocol = build_protocol_from_config(case_config)
        split_records = materialize_split_signals(protocol)
        root_split_maps = build_input_split_results(split_records, channel_ids=list(state.dag_state.channels))
        labels_by_split = build_split_labels(split_records)
        protocol_summary = {
            **summarize_protocol(protocol),
            **summarize_split_records(split_records),
        }
        split_manifest = export_split_manifest(protocol)
        split_summary_by_label_domain = summarize_split_ids_by_label_domain(protocol)
    else:
        raise ValueError("evaluate_full_dag_leaves requires either case_config or split_records.")

    replayed_state = replay_state_on_split_maps(
        state,
        root_split_maps=root_split_maps,
        labels_by_split=labels_by_split,
        split_names=("train", "val", "test"),
    )
    datasets, failures = extract_leaf_datasets(
        replayed_state,
        labels_by_split=labels_by_split,
        split_names=("train", "val", "test"),
    )
    ml_results = shallow_ml_agent(
        datasets,
        algorithm=algorithm,
        ensemble_method=ensemble_method,
        cv_folds=cv_folds,
        candidate_algorithms=candidate_algorithms,
        parallel_workers=parallel_workers,
    )
    ml_results["dag_summary"] = summarize_dag_state(replayed_state)
    ml_results["protocol_summary"] = protocol_summary
    if failures:
        ml_results["leaf_failures"] = failures
        existing = list(ml_results.get("node_level_results") or [])
        existing.extend(failures)
        ml_results["node_level_results"] = existing
    best_single_leaf_id = str((ml_results.get("final_selection") or {}).get("best_single_leaf", "") or "")
    best_single_metrics = dict((ml_results.get("final_selection") or {}).get("best_single_leaf_metrics") or {})
    normalized_selection = {
        "best_single_leaf": {"leaf_id": best_single_leaf_id, **best_single_metrics} if best_single_leaf_id else {},
        "weighted_ensemble": {
            "metrics": dict(ml_results.get("weighted_ensemble_metrics") or {}),
        },
        "final_choice": {
            "strategy": str((ml_results.get("final_selection") or {}).get("final_choice", "")),
        },
        "selection_basis": str((ml_results.get("final_selection") or {}).get("selection_basis", "")),
    }
    selection_predictions = dict(ml_results.get("selection_predictions") or {})
    return {
        "state": replayed_state,
        "datasets": datasets,
        "ml_results": ml_results,
        "leaf_metrics": list(ml_results.get("node_level_results") or []),
        "final_selection": normalized_selection,
        "selection_predictions": selection_predictions,
        "metrics_markdown": str(ml_results.get("metrics_markdown") or ""),
        "dag_summary": ml_results["dag_summary"],
        "protocol_summary": protocol_summary,
        "split_manifest": split_manifest,
        "split_summary_by_label_domain": split_summary_by_label_domain,
    }
