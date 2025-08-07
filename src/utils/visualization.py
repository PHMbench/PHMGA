from __future__ import annotations

import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# --- Keep the old function for now, or remove it if no longer needed ---
# from src.states.phm_states import PHMState, DAGState, InputData
# def visualize_dag_feature_evolution_umap(state: PHMState) -> None:
#     ... (old implementation)


def visualize_datasets_umap(
    datasets: Dict[str, Dict[str, Any]], 
    save_dir: str = "save/dataset_umaps",
    plot_train: bool = False,
    show_title: bool = False
) -> None:
    """
    Generates and saves UMAP visualizations for datasets.

    By default, it plots only the test set. The training set can be included
    optionally. The UMAP embedding is always learned on all available data
    (train + test) to ensure a consistent projection space.

    Args:
        datasets: Dictionary of datasets from dataset_preparer_agent.
        save_dir: Directory to save the output PNG files.
        plot_train: If True, plots the training data points. Defaults to False.
        show_title: If True, adds a title to the plot. Defaults to False.
    """
    print(f"--- Generating UMAP visualizations for {len(datasets)} datasets ---")
    os.makedirs(save_dir, exist_ok=True)

    for node_id, data in datasets.items():
        print(f"Processing: {node_id}")

        # --- 1. Get Train and Test data ---
        X_train, y_train = data.get("X_train"), data.get("y_train")
        X_test, y_test = data.get("X_test"), data.get("y_test")
        # y_train, y_test = int(y_train), int(y_test)
        # Basic validation
        if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray) or X_test.shape[0] == 0:
            print(f"  Skipping {node_id}: Test set is empty or invalid.")
            continue
        
        n_train = X_train.shape[0] if isinstance(X_train, np.ndarray) else 0
        n_test = X_test.shape[0]

        # Combine features for a unified UMAP transformation
        X_combined = np.vstack((X_train, X_test)) if n_train > 0 else X_test

        if X_combined.shape[0] < 2:
            print(f"  Skipping {node_id}: Not enough samples ({X_combined.shape[0]}) for UMAP.")
            continue

        # --- 2. Preprocess and run UMAP on combined data ---
        X_scaled = StandardScaler().fit_transform(X_combined)
        
        n_neighbors = min(15, X_scaled.shape[0] - 1)
        if n_neighbors < 2:
             print(f"  Skipping {node_id}: Too few samples to determine valid n_neighbors > 1.")
             continue

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(X_scaled)

        # Split the embedding back into train and test sets
        embedding_train = embedding[:n_train] if n_train > 0 else np.array([])
        embedding_test = embedding[n_train:]

        # --- 3. Plot the results ---
        plt.figure(figsize=(12, 8))

        y_combined = np.concatenate((y_train, y_test)) if n_train > 0 and plot_train else y_test
        unique_labels = np.unique(y_combined)
        norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))
        cmap = plt.get_cmap('Spectral')

        # Plot training data (circles) if requested
        if plot_train and n_train > 0:
            plt.scatter(
                embedding_train[:, 0], embedding_train[:, 1], 
                c=y_train, cmap=cmap, norm=norm, s=50, marker='o'
            )
        
        # Plot test data (crosses)
        plt.scatter(
            embedding_test[:, 0], embedding_test[:, 1], 
            c=y_test, cmap=cmap, norm=norm, s=60, marker='X'
        )
        
        if show_title:
            plt.title(f"UMAP Projection of '{node_id}' Features", fontsize=16)
        
        plt.xlabel("D1")
        plt.ylabel("D2")
        
        # --- Create a dynamic legend ---
        if plot_train and n_train > 0:
            marker_handles = []
            marker_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='grey', markersize=10))
            marker_handles.append(plt.Line2D([0], [0], marker='X', color='w', label='Test', markerfacecolor='grey', markersize=10))
        
            marker_legend = plt.legend(handles=marker_handles, title="Dataset")
            plt.gca().add_artist(marker_legend)

        # FIX: Convert each label to an integer before creating the legend string.
        color_handles = [mpatches.Patch(color=cmap(norm(label)), label=str(int(label))) for label in unique_labels]
        plt.legend(handles=color_handles, title="Classes", loc='upper right')

        plt.grid(True)

        # --- 4. Save the figure ---
        safe_filename = node_id.replace("/", "_").replace("\\", "_")
        save_path = os.path.join(save_dir, f"{safe_filename}_umap.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to {save_path}")

    print(f"\n--- UMAP visualization process finished. ---")


if __name__ == "__main__":
    print("--- Running Visualization Demo ---")

    # 1. Create a mock 'datasets' dictionary, simulating output from dataset_preparer_agent
    mock_datasets = {
        "fft_kurtosis_ch1": {
            "X_train": np.random.rand(50, 10),  # 50 samples, 10 features
            "y_train": np.zeros(50),             # All class 0
            "X_test": np.random.rand(50, 10) + 0.5, # 50 samples, 10 features (shifted)
            "y_test": np.ones(50),               # All class 1
        },
        "hilbert_rms_ch2": {
            "X_train": np.random.rand(60, 5),   # 60 samples, 5 features
            "y_train": np.array([0] * 30 + [1] * 30), # Mixed classes 0 and 1
            "X_test": np.random.rand(40, 5),
            "y_test": np.array([2] * 20 + [3] * 20), # Classes 2 and 3
        },
        "empty_node_ch3": { # A node with not enough data
            "X_train": np.random.rand(1, 20),
            "y_train": np.array([0]),
            "X_test": np.array([]).reshape(0, 20), # Empty test set
            "y_test": np.array([]),
        }
    }

    # 2. Call the new visualization function with the mock data
    visualize_datasets_umap(mock_datasets)

    print("\n--- Demo Finished ---")
