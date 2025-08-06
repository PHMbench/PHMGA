import os
from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import umap

from src.states.phm_states import PHMState, DAGState, InputData


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

        # --- CRITICAL FIX: Reshape data for UMAP ---
        # UMAP expects a 2D array of shape (n_samples, n_features).
        # We reshape the data, treating each sample's features as a flattened vector.
        if data.ndim > 2:
            n_samples = data.shape[0]
            # Flatten all dimensions except the first (samples)
            data = data.reshape(n_samples, -1)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = data.shape[0]
        if n_samples < 2:
            print(f"Node {node_id} has insufficient samples ({n_samples}) for UMAP; skipping.")
            continue

        # n_neighbors must be smaller than n_samples.
        # We also need n_neighbors > 1 for the algorithm to work.
        n_neighbors = min(15, n_samples - 1)
        if n_neighbors <= 1:
            print(f"Node {node_id} has too few samples ({n_samples}) to determine valid n_neighbors > 1; skipping.")
            continue

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

    # 1. Create a mock DAGState with a more realistic number of samples and evolved nodes
    instruction = "Analyze the bearing signals from multiple channels for potential faults."
    
    initial_nodes = {}
    initial_leaves = []
    channels = ["ch1", "ch2"]
    n_samples_ref = 50
    n_samples_tst = 50
    n_total_samples = n_samples_ref + n_samples_tst

    for channel_name in channels:
        node = InputData(
            node_id=channel_name,
            results={
                "ref": np.random.randn(n_samples_ref, 1024, 1),
                "tst": np.random.randn(n_samples_tst, 1024, 1) * 1.5
            },
            parents=[],
            shape=(n_total_samples, 1024, 1),
            meta={"channel": channel_name}
        )
        initial_nodes[channel_name] = node
        initial_leaves.append(channel_name)

    # --- Simulate an evolved node (e.g., after FFT) ---
    fft_node = InputData( # Using InputData for simplicity in this mock
        node_id="fft_01_ch1",
        results={
            "ref": np.random.rand(n_samples_ref, 513, 1), # Shape after rfft
            "tst": np.random.rand(n_samples_tst, 513, 1)
        },
        parents=["ch1"],
        shape=(n_total_samples, 513, 1),
        meta={}
    )
    initial_nodes["fft_01_ch1"] = fft_node

    # --- Simulate a final feature node (e.g., after mean) ---
    feature_node = InputData(
        node_id="mean_01_fft_01_ch1",
        results={
            "ref": np.random.rand(n_samples_ref, 1), # Shape after aggregation
            "tst": np.random.rand(n_samples_tst, 1) * 1.2
        },
        parents=["fft_01_ch1"],
        shape=(n_total_samples, 1),
        meta={}
    )
    initial_nodes["mean_01_fft_01_ch1"] = feature_node


    dag = DAGState(
        user_instruction=instruction, 
        channels=channels, 
        nodes=initial_nodes, 
        leaves=["fft_01_ch1", "ch2", "mean_01_fft_01_ch1"] # Update leaves
    )
    
    # 2. Create mock datasets that would be generated by the dataset_preparer_agent
    mock_datasets = {
        "feature_node_dataset": {
            "y_train": np.array([0] * n_samples_ref), # Labels for reference data
            "y_test": np.array([1] * n_samples_tst)  # Labels for test data
        }
    }

    # 3. Assemble the full PHMState object
    state = PHMState(
        user_instruction=instruction,
        dag_state=dag,
        datasets=mock_datasets,
        # --- FIX: Provide all required fields for InputData validation ---
        reference_signal=InputData(node_id="ref_placeholder", results={}, parents=[], shape=()),
        test_signal=InputData(node_id="tst_placeholder", results={}, parents=[], shape=())
    )

    # 4. Call the visualization function
    visualize_dag_feature_evolution_umap(state)
    
    print("\n--- Demo Finished ---")
    print("UMAP visualizations saved to 'save/feature_evolution_umap'.")
