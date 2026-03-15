"""Shallow ML baselines for the rebuilt `ml` path."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

from src.data.dataset_preparer import DatasetView


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _build_estimator(algorithm: str, max_iter: int, random_state: int) -> Any:
    normalized = algorithm.strip().lower()
    if normalized == "logistic_regression":
        return LogisticRegression(max_iter=max_iter, random_state=random_state)
    if normalized == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    if normalized == "svm":
        return SVC(kernel="rbf", probability=False, random_state=random_state)
    raise ValueError(f"Unsupported shallow ML algorithm: {algorithm}")


def _importance_payload(estimator: Any, feature_count: int) -> Dict[int, float]:
    if hasattr(estimator, "coef_"):
        weights = np.asarray(estimator.coef_[0], dtype=float)
        return {index: float(abs(weight)) for index, weight in enumerate(weights)}
    if hasattr(estimator, "feature_importances_"):
        weights = np.asarray(estimator.feature_importances_, dtype=float)
        return {index: float(weight) for index, weight in enumerate(weights)}
    return {index: 0.0 for index in range(feature_count)}


def run_shallow_ml_baseline(
    dataset_views: Dict[str, DatasetView],
    algorithm: str,
    *,
    max_iter: int = 200,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Train and evaluate a shallow ML baseline on prepared dataset views."""

    estimator = _build_estimator(algorithm, max_iter=max_iter, random_state=random_state)
    train_view = dataset_views["train"]
    estimator.fit(train_view.X, train_view.y)

    predictions: Dict[str, list[dict[str, object]]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        view = dataset_views[split_name]
        preds = estimator.predict(view.X)
        metrics[split_name] = _compute_metrics(view.y, preds)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(prediction)}
            for sample_id, prediction in zip(view.sample_ids, preds)
        ]

    importance = _importance_payload(estimator, train_view.X.shape[1])
    return {
        "algorithm": algorithm,
        "metrics": metrics,
        "predictions": predictions,
        "importance_by_index": importance,
    }
