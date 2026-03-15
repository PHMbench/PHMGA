"""Optional similarity analysis artifacts for downstream reports.

This module keeps the useful comparison logic from the old `inquirer_agent`,
but rewrites it around the rebuilt repo's canonical `train/val/test` splits.
It produces side artifacts only; it does not participate in workflow-state
transitions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.data.dataset_preparer import DatasetView


def _pairwise_centroid_distances(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if x.size == 0 or y.size == 0:
        return {"l2": 0.0, "cosine": 0.0}
    l2 = float(np.linalg.norm(x - y))
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(x, y) / (x_norm * y_norm))
    return {"l2": l2, "cosine": cosine}


def build_similarity_artifacts(dataset_views: Dict[str, DatasetView]) -> Dict[str, Any]:
    """Build split-aware similarity summaries from dataset views."""

    artifacts: Dict[str, Any] = {"split_sizes": {}}
    for split_name, view in dataset_views.items():
        artifacts["split_sizes"][split_name] = int(view.X.shape[0])

    train_view = dataset_views.get("train")
    if train_view is None or train_view.X.size == 0:
        artifacts["class_centroid_similarity"] = {}
        return artifacts

    class_centroids: Dict[str, list[float]] = {}
    for label in sorted(set(train_view.y.tolist())):
        mask = train_view.y == label
        class_centroids[str(label)] = train_view.X[mask].mean(axis=0).tolist()
    artifacts["class_centroids"] = class_centroids

    centroid_similarity: Dict[str, Dict[str, float]] = {}
    centroid_vectors = {label: np.asarray(values, dtype=float) for label, values in class_centroids.items()}
    labels = sorted(centroid_vectors)
    for index, left_label in enumerate(labels):
        for right_label in labels[index + 1 :]:
            key = f"{left_label}__vs__{right_label}"
            centroid_similarity[key] = _pairwise_centroid_distances(
                centroid_vectors[left_label],
                centroid_vectors[right_label],
            )
    artifacts["class_centroid_similarity"] = centroid_similarity

    test_view = dataset_views.get("test")
    if test_view is not None and test_view.X.size and centroid_vectors:
        mean_train_centroid = np.asarray(list(class_centroids.values()), dtype=float).mean(axis=0)
        mean_test = test_view.X.mean(axis=0)
        artifacts["test_to_train_mean_similarity"] = _pairwise_centroid_distances(mean_train_centroid, mean_test)

    return artifacts
