from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np

from src.states.phm_states import PHMState, ProcessedData, DataSetNode, InputData


def _find_root_labels(node_id: str, all_nodes: Dict[str, InputData | ProcessedData]) -> Dict[str, Any]:
    """
    Traverse up the DAG from a given node to find its root and return the labels
    stored in the root's metadata.
    """
    current_node = all_nodes.get(node_id)
    if not current_node:
        return {}

    # Keep moving to the parent until a node with no parents (the root) is found.
    # This assumes a single-parent lineage for processed nodes, which is typical.
    while current_node.parents:
        parent_id = current_node.parents[0]
        parent_node = all_nodes.get(parent_id)
        if not parent_node:
            # This should not happen in a well-formed DAG
            return {}
        current_node = parent_node

    # Once at the root node, extract the 'labels' dictionary from its metadata.
    return current_node.meta.get("labels", {})


def _build_dataset_from_features(
    feature_path: str, labels_map: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds a dataset (features and labels) by matching sample IDs from a feature
    file with a provided labels dictionary.
    """
    if not feature_path or not os.path.exists(feature_path) or not labels_map:
        return np.array([]), np.array([])

    features_list = []
    labels_list = []

    with np.load(feature_path) as data:
        # Iterate through the sample IDs found in the feature file
        for sample_id in data.files:
            if sample_id in labels_map:
                feature = data[sample_id]
                features_list.append(feature.reshape(1, -1))
                labels_list.append(labels_map[sample_id])
            else:
                print(f"Warning: Sample ID '{sample_id}' found in feature file but not in labels map. Skipping.")

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
    datasets: Dict[str, Dict[str, Any]] = {}
    tracker = state.tracker()
    all_nodes = state.dag_state.nodes

    for node_id, node in list(all_nodes.items()):
        if getattr(node, "stage", None) != stage:
            continue

        # For each processed node, find its corresponding true labels from its root.
        labels_map = _find_root_labels(node_id, all_nodes)
        if not labels_map:
            print(f"Warning: Could not find root labels for node {node_id}. Skipping dataset creation.")
            continue

        saved = node.meta.get("saved", {})
        ref_path = saved.get("ref_path")
        tst_path = saved.get("tst_path")

        # Build training and test sets using the found labels
        X_train, y_train = _build_dataset_from_features(ref_path, labels_map)
        X_test, y_test = _build_dataset_from_features(tst_path, labels_map)

        if X_train.size == 0 and X_test.size == 0:
            continue
        
        datasets[node_id] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "origin_node": node_id,
        }
        
        # ds_node = DataSetNode(
        #     node_id=f"ds_{node_id}",
        #     parents=[node_id],
        #     shape=X_train.shape if X_train.size else X_test.shape,
        #     meta={"origin_node": node_id, "channel": node.meta.get("channel")},
        # )
        # tracker.add_node(ds_node)

    return {"datasets": datasets, "n_nodes": len(datasets)}


# The if __name__ == "__main__": block is now outdated and would need to be
# updated to reflect this new logic.
if __name__ == "__main__":
    print("The test case in __main__ needs to be updated for the new agent logic.")
