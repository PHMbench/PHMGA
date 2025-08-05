import os
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import umap

from src.states.phm_states import PHMState, DAGState


def visualize_dag_feature_evolution_umap(dag: DAGState, state: PHMState, y_labels: List[Any]) -> None:
    """Generate UMAP projections of features at each node in the DAG.

    Parameters
    ----------
    dag : DAGState
        The DAG structure whose nodes contain computed results.
    state : PHMState
        Full state object used to access the tracker for ordering.
    y_labels : List[Any]
        Labels used to color the samples in the scatter plots.
    """
    tracker = state.tracker()
    try:
        node_ids = list(nx.topological_sort(tracker.g))
    except Exception:
        node_ids = list(dag.nodes.keys())

    output_dir = os.path.join("save", "feature_evolution_umap")
    os.makedirs(output_dir, exist_ok=True)

    for idx, node_id in enumerate(node_ids, start=1):
        node = dag.nodes.get(node_id)
        if node is None:
            continue

        data = node.results
        if isinstance(data, dict):
            arrays: List[np.ndarray] = []
            for val in data.values():
                if isinstance(val, np.ndarray):
                    arrays.append(val)
                elif isinstance(val, dict):
                    arrays.extend([v for v in val.values() if isinstance(v, np.ndarray)])
            if not arrays:
                print(f"Node {node_id} has no numeric results; skipping.")
                continue
            try:
                data = np.concatenate(arrays, axis=0)
            except Exception:
                print(f"Failed to concatenate results for node {node_id}; skipping.")
                continue
        else:
            data = np.asarray(data)

        if data.size == 0:
            print(f"Node {node_id} produced empty results; skipping.")
            continue

        if np.iscomplexobj(data):
            data = np.abs(data)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = data.shape[0]
        if n_samples < 2:
            print(f"Node {node_id} has insufficient samples ({n_samples}); skipping.")
            continue

        n_neighbors = min(15, n_samples - 1)
        try:
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            embedding = reducer.fit_transform(data)
        except Exception as exc:
            print(f"UMAP failed for node {node_id}: {exc}")
            continue

        plt.figure()
        try:
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_labels[:n_samples], palette="viridis", s=40)
            plt.legend()
        except Exception:
            plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title(f"UMAP Projection of Node: {node_id}")
        plt.tight_layout()
        file_name = f"{idx:02d}_{node_id}.png"
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()
