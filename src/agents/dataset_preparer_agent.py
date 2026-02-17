from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

from src.states.phm_states import PHMState, ProcessedData, DataSetNode, InputData


def _find_root_label_maps(
    node_id: str,
    all_nodes: Dict[str, InputData | ProcessedData],
    *,
    max_hops: int = 1024,
    error_sink: list[str] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Traverse up the DAG from a given node to find its root and return
    (labels_ref, labels_tst) stored in the root's metadata.
    """
    current_node = all_nodes.get(node_id)
    if not current_node:
        return {}, {}

    visited: set[str] = set()
    hops = 0

    # Keep moving to the parent until a node with no parents (the root) is found.
    # This assumes a single-parent lineage for processed nodes, which is typical.
    while current_node.parents:
        hops += 1
        if hops > max_hops:
            if error_sink is not None:
                error_sink.append(
                    f"_find_root_label_maps exceeded max_hops={max_hops} while tracing node '{node_id}'."
                )
            return {}, {}
        parent_id = current_node.parents[0]
        if parent_id in visited:
            if error_sink is not None:
                error_sink.append(
                    f"_find_root_label_maps detected parent cycle while tracing node '{node_id}' (at '{parent_id}')."
                )
            return {}, {}
        visited.add(parent_id)
        parent_node = all_nodes.get(parent_id)
        if not parent_node:
            # This should not happen in a well-formed DAG
            if error_sink is not None:
                error_sink.append(
                    f"_find_root_label_maps missing parent '{parent_id}' while tracing node '{node_id}'."
                )
            return {}, {}
        current_node = parent_node

    # Once at the root node, extract the 'labels' dictionary from its metadata.
    labels_ref = current_node.meta.get("labels_ref", {}) or {}
    labels_tst = current_node.meta.get("labels_tst", {}) or {}
    if not labels_ref and not labels_tst:
        # Backward-compatible fallback (discouraged; may include leakage).
        labels = current_node.meta.get("labels", {}) or {}
        labels_ref = labels
        labels_tst = labels
    return labels_ref, labels_tst


def _build_dataset_from_features(feature_path: str, labels_map: Dict[str, Any], *, flatten: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a dataset (features and labels) by matching sample IDs from a feature
    file with a provided labels dictionary.
    """
    if not feature_path or not os.path.exists(feature_path) or not labels_map:
        return np.array([]), np.array([])

    features_list: list[np.ndarray] = []
    labels_list: list[Any] = []

    data = np.load(feature_path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        # Support legacy single-array saves (.npy). We treat the whole file as one sample.
        # If flatten is enabled and the array is 1D, treat each element as one row so
        # downstream models can still train.
        sample_ids = list(labels_map.keys())
        sample_id = sample_ids[0] if sample_ids else "sample"
        if flatten and data.ndim == 1:
            X = data.reshape(-1, 1)
            y = np.full((X.shape[0],), labels_map.get(sample_id))
            return X, y
        features_list.append(data.reshape(1, -1))
        labels_list.append(labels_map.get(sample_id))
    else:
        # .npz: Iterate through the sample IDs found in the archive
        with data as npz:
            for sample_id in npz.files:
                if sample_id in labels_map:
                    feature = npz[sample_id]
                    features_list.append(feature.reshape(1, -1))
                    labels_list.append(labels_map[sample_id])
                else:
                    print(
                        f"Warning: Sample ID '{sample_id}' found in feature file but not in labels map. Skipping."
                    )

    if not features_list:
        return np.array([]), np.array([])

    X = np.vstack(features_list)
    y = np.array(labels_list)
    return X, y


def dataset_preparer_agent(state: PHMState, *, config: Dict | None = None) -> Dict:
    """
    Gathers features and assembles datasets using true labels found by traversing
    the DAG back to the root nodes.
    """
    cfg = config or {}
    stage = cfg.get("stage", "processed")
    flatten = bool(cfg.get("flatten", False))
    datasets: Dict[str, Dict[str, Any]] = {}
    tracker = state.tracker()
    all_nodes = state.dag_state.nodes

    for node_id, node in list(all_nodes.items()):
        if getattr(node, "stage", None) != stage:
            continue

        # For each processed node, find its corresponding true labels from its root.
        labels_ref, labels_tst = _find_root_label_maps(
            node_id,
            all_nodes,
            error_sink=state.dag_state.error_log,
        )
        if not labels_ref and not labels_tst:
            print(f"Warning: Could not find root labels for node {node_id}. Skipping dataset creation.")
            continue

        saved = node.meta.get("saved", {})
        ref_path = saved.get("ref_path")
        tst_path = saved.get("tst_path")

        # Build training and test sets using the found labels (separate maps).
        X_train, y_train = _build_dataset_from_features(ref_path, labels_ref, flatten=flatten)

        allow_test = bool(getattr(state, "allow_test_labels_for_reporting", False))
        X_test, y_test = _build_dataset_from_features(tst_path, labels_tst, flatten=flatten) if allow_test else (np.array([]), np.array([]))

        if X_train.size == 0 and X_test.size == 0:
            continue
        
        datasets[node_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "origin_node": node_id,
        }
        
        ds_node = DataSetNode(
            node_id=f"ds_{node_id}",
            parents=[node_id],
            shape=X_train.shape if X_train.size else X_test.shape,
            meta={
                "origin_node": node_id,
                "channel": node.meta.get("channel"),
                "n_train": int(X_train.shape[0]) if X_train.size else 0,
                "n_test": int(X_test.shape[0]) if X_test.size else 0,
            },
        )
        tracker.add_node(ds_node)

    return {"datasets": datasets, "n_nodes": len(datasets)}


# The if __name__ == "__main__": block is now outdated and would need to be
# updated to reflect this new logic.
if __name__ == "__main__":
    print("The test case in __main__ needs to be updated for the new agent logic.")
