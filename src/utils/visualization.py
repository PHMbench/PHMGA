from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches


def _subsample_by_class(
    X: np.ndarray, y: np.ndarray, source: np.ndarray, max_per_class: int = 1000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsamples data to a maximum number of samples per class."""
    df = pd.DataFrame(X)
    df['label'] = y
    df['source'] = source

    # Group by label and sample, then combine
    sampled_df = df.groupby('label').apply(
        lambda grp: grp.sample(n=min(len(grp), max_per_class), random_state=42)
    ).reset_index(drop=True)
    
    X_sampled = sampled_df.drop(columns=['label', 'source']).values
    y_sampled = sampled_df['label'].values
    source_sampled = sampled_df['source'].values
    
    return X_sampled, y_sampled, source_sampled


def visualize_datasets_umap(
    datasets: Dict[str, Dict[str, Any]], 
    save_dir: str = "save/dataset_umaps",
    plot_train: bool = False,
    show_title: bool = False,
    max_samples_per_class: int = 1000
) -> None:
    """
    Generates and saves UMAP visualizations, subsampling if data is large.

    Args:
        datasets: Dictionary of datasets from dataset_preparer_agent.
        save_dir: Directory to save the output PNG files.
        plot_train: If True, plots the training data points. Defaults to False.
        show_title: If True, adds a title to the plot. Defaults to False.
        max_samples_per_class: Max samples per class to use for UMAP. Defaults to 1000.
    """
    print(f"--- Generating UMAP visualizations for {len(datasets)} datasets ---")
    os.makedirs(save_dir, exist_ok=True)

    for node_id, data in datasets.items():
        print(f"Processing: {node_id}")

        # TODO: Implement dimensionality reduction for high-dimensional data
        if data["X_train"].shape[1] > 10000:
            data["X_train"] = data["X_train"][:, :10000]
        if data["X_test"].shape[1] > 10000:
            data["X_test"] = data["X_test"][:, :10000]
        # continue when data have nan
        if np.isnan(data["X_train"]).any() or np.isnan(data["X_test"]).any():
            print(f"  Skipping {node_id}: Data contains NaN values.")
            continue

        X_train, y_train = data.get("X_train"), data.get("y_train")
        X_test, y_test = data.get("X_test"), data.get("y_test")

        if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray) or X_test.shape[0] == 0:
            print(f"  Skipping {node_id}: Test set is empty or invalid.")
            continue
        
        n_train = X_train.shape[0] if isinstance(X_train, np.ndarray) and plot_train else 0
        n_test = X_test.shape[0]

        # --- 1. Combine data and create source labels (0=train, 1=test) ---
        X_combined = np.vstack((X_train, X_test)) if n_train > 0 else X_test
        y_combined = np.concatenate((y_train, y_test)) if n_train > 0 else y_test
        source_labels = np.concatenate((np.zeros(n_train), np.ones(n_test))) if n_train > 0 else np.ones(n_test)

        # --- 2. Subsample if necessary ---
        if X_combined.shape[0] > max_samples_per_class:
            print(f"  Subsampling data from {X_combined.shape[0]} to max {max_samples_per_class} per class.")
            X_combined, y_combined, source_labels = _subsample_by_class(
                X_combined, y_combined, source_labels, max_samples_per_class
            )
            print(f"  New sample size: {X_combined.shape[0]}")

        if X_combined.shape[0] < 2:
            print(f"  Skipping {node_id}: Not enough samples ({X_combined.shape[0]}) for UMAP.")
            continue

        # --- 3. Preprocess and run UMAP ---
        X_scaled = StandardScaler().fit_transform(X_combined)
        n_neighbors = min(15, X_scaled.shape[0] - 1)
        if n_neighbors < 2:
             print(f"  Skipping {node_id}: Too few samples for valid n_neighbors.")
             continue

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(X_scaled)

        # --- 4. Plot the results ---
        plt.figure(figsize=(12, 8))
        
        unique_labels = np.unique(y_combined)
        norm = plt.Normalize(vmin=np.min(unique_labels), vmax=np.max(unique_labels))
        cmap = plt.get_cmap('Spectral')

        # Separate embedding back into train and test based on source labels
        is_train = (source_labels == 0)
        is_test = (source_labels == 1)

        if plot_train and np.any(is_train):
            plt.scatter(
                embedding[is_train, 0], embedding[is_train, 1], 
                c=y_combined[is_train], cmap=cmap, norm=norm, s=50, marker='o'
            )
        
        plt.scatter(
            embedding[is_test, 0], embedding[is_test, 1], 
            c=y_combined[is_test], cmap=cmap, norm=norm, s=60, marker='X'
        )
        
        if show_title:
            plt.title(f"UMAP Projection of '{node_id}' Features", fontsize=16)
        
        plt.xlabel("D1")
        plt.ylabel("D2")
        
        marker_handles = []
        if plot_train and np.any(is_train):
            marker_handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='grey', markersize=10))
        marker_handles.append(plt.Line2D([0], [0], marker='X', color='w', label='Test', markerfacecolor='grey', markersize=10))
        
        if marker_handles:
            marker_legend = plt.legend(handles=marker_handles, title="Dataset")
            plt.gca().add_artist(marker_legend)

        color_handles = [mpatches.Patch(color=cmap(norm(label)), label=str(int(label))) for label in unique_labels]
        plt.legend(handles=color_handles, title="Classes", loc='upper right')

        plt.grid(True)

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
