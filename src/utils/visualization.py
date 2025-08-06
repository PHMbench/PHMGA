import os
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import umap

from src.states.phm_states import PHMState, DAGState


def visualize_dag_feature_evolution_umap(state: PHMState) -> None:
    """
    Generate UMAP projections of features at each node in the DAG using data from the state.

    Parameters
    ----------
    state : PHMState
        The full state object containing the DAG, datasets, and other necessary info.
    """
    dag = state.dag_state
    if not dag or not dag.nodes:
        print("DAG state is empty. Skipping UMAP visualization.")
        return

    # --- Extract labels from the state's datasets ---
    y_labels = []
    if state.datasets:
        # Get the first available dataset to extract labels
        first_dataset_key = next(iter(state.datasets))
        dataset = state.datasets[first_dataset_key]
        y_train = dataset.get('y_train')
        y_test = dataset.get('y_test')
        
        if y_train is not None:
            y_labels.extend(list(y_train))
        if y_test is not None:
            y_labels.extend(list(y_test))

    if not y_labels:
        print("Could not find labels in state.datasets. Plots will be uncolored.")

    # --- Get node order ---
    try:
        # This part is simplified as tracker might not be a standard part of the state
        node_ids = list(dag.nodes.keys()) # Simple iteration if topological sort fails or isn't needed
    except Exception:
        node_ids = list(dag.nodes.keys())

    output_dir = os.path.join("save", "feature_evolution_umap")
    os.makedirs(output_dir, exist_ok=True)

    for idx, node_id in enumerate(node_ids, start=1):
        node = dag.nodes.get(node_id)
        if node is None:
            continue

        # MODIFIED: Access results from the node, which should contain 'ref' and 'tst' keys
        results_dict = node.results
        if not isinstance(results_dict, dict) or not results_dict:
            print(f"Node {node_id} has no dictionary results; skipping.")
            continue
        
        # Concatenate 'ref' and 'tst' data if they exist
        ref_data = results_dict.get("ref")
        tst_data = results_dict.get("tst")
        
        all_data = []
        if isinstance(ref_data, np.ndarray):
            all_data.append(ref_data)
        if isinstance(tst_data, np.ndarray):
            all_data.append(tst_data)

        if not all_data:
            print(f"Node {node_id} has no 'ref' or 'tst' numpy arrays in results; skipping.")
            continue
        
        try:
            data = np.concatenate(all_data, axis=0)
        except ValueError as e:
            print(f"Failed to concatenate results for node {node_id} due to shape mismatch: {e}; skipping.")
            continue

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
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=y_labels[:n_samples] if y_labels else None, palette="viridis", s=40)
            plt.legend()
        except Exception:
            plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.title(f"UMAP Projection of Node: {node_id}")
        plt.tight_layout()
        file_name = f"{idx:02d}_{node_id}.png"
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()


if __name__ == "__main__":
    # --- Optimized Test Demo ---
    print("--- Running Optimized Visualization Demo ---")

    # 1. Create a mock DAGState with two nodes representing a simple pipeline
    # --- 模拟一个更真实的初始状态 ---
    instruction = "Analyze the bearing signals from multiple channels for potential faults."
    
    initial_nodes = {}
    initial_leaves = []
    channels = ["ch1", "ch2", "ch3"]
    for channel_name in channels:
        node = InputData(
            node_id=channel_name,
            results={
                "ref": np.random.randn(1, 1024, 1),
                "tst": np.random.randn(1, 1024, 1) * 1.5
            },
            parents=[],
            shape=(1, 1024, 1),
            meta={"channel": channel_name}
        )
        initial_nodes[channel_name] = node
        initial_leaves.append(channel_name)

    dag = DAGState(
        user_instruction=instruction, 
        channels=channels, 
        nodes=initial_nodes, 
        leaves=initial_leaves
    )
    
    state = PHMState(
        user_instruction=instruction, 
        reference_signal=initial_nodes["ch1"], 
        test_signal=initial_nodes["ch1"], 
        dag_state=dag
    )

    # 2. Create mock datasets that would be generated by the dataset_preparer_agent
    mock_datasets = {
        "feature_node_dataset": {
            "X_train": np.random.rand(50, 5),
            "y_train": np.array([0] * 25 + [1] * 25), # 2 classes for reference data
            "X_test": np.random.rand(25, 5),
            "y_test": np.array([0] * 10 + [1] * 15)  # 2 classes for test data
        }
    }

    # 3. Assemble the full PHMState object
    mock_state = PHMState(
        dag_state=mock_dag_state,
        datasets=mock_datasets,
        user_instruction="A mock state for visualization testing."
        # Other state fields can be added if needed
    )

    # 4. Call the simplified function
    visualize_dag_feature_evolution_umap(mock_state)
    
    print("\n--- Demo Finished ---")
    print("UMAP visualizations saved to 'save/feature_evolution_umap'.")
