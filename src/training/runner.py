"""Minimal runners for the ``ml`` and ``torch`` graph paths."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.bridge import FeaturePipelinePlan, ModelBuildPlan
from src.data import SignalRecord
from src.model import build_feature_matrix
from src.operators import OperatorCatalog


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the two primary metrics used by the rebuilt repo."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def run_ml_pipeline(
    plan: FeaturePipelinePlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    max_iter: int = 200,
) -> Dict[str, object]:
    """Run the lightweight ML baseline on bridge-generated feature matrices."""
    matrices = build_feature_matrix(plan, split_records, catalog)
    clf = LogisticRegression(max_iter=max_iter, random_state=0)
    clf.fit(matrices["train"]["X"], matrices["train"]["y"])
    predictions: Dict[str, list[dict[str, object]]] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        preds = clf.predict(matrices[split_name]["X"])
        metrics[split_name] = _compute_metrics(matrices[split_name]["y"], preds)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(pred)}
            for sample_id, pred in zip(matrices[split_name]["sample_ids"], preds)
        ]
    importance = {
        spec.feature_node_id: float(abs(weight))
        for spec, weight in zip(plan.feature_specs, clf.coef_[0])
    }
    return {
        "feature_pipeline": plan.model_dump(),
        "metrics": metrics,
        "predictions": predictions,
        "importance": importance,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax used by the fallback trainable path."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Expand integer labels into one-hot rows for cross-entropy training."""
    eye = np.eye(num_classes, dtype=float)
    return eye[labels]


def run_torch_pipeline(
    plan: ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    epochs: int = 12,
    learning_rate: float = 0.2,
) -> Dict[str, object]:
    """Run the current trainable path implementation.

    Despite the path name, the rebuilt repository currently uses a NumPy
    fallback here so smoke tests can validate the full artifact contract
    without requiring a PyTorch runtime.
    """
    matrices = build_feature_matrix(plan, split_records, catalog)
    x_train = matrices["train"]["X"]
    y_train = matrices["train"]["y"]
    num_classes = int(np.max(y_train)) + 1
    weights = np.zeros((x_train.shape[1], num_classes), dtype=float)
    bias = np.zeros((num_classes,), dtype=float)
    curves: list[dict[str, float]] = []
    for epoch in range(epochs):
        # This is a minimal linear classifier trained by gradient descent. It
        # exists to exercise the trainable artifact path, not to be a final
        # research-grade deep model.
        logits = x_train @ weights + bias
        probs = _softmax(logits)
        targets = _one_hot(y_train, num_classes)
        error = probs - targets
        grad_w = x_train.T @ error / x_train.shape[0]
        grad_b = error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        preds = np.argmax(probs, axis=1)
        metrics = _compute_metrics(y_train, preds)
        loss = float(-np.mean(np.sum(targets * np.log(np.clip(probs, 1e-8, 1.0)), axis=1)))
        curves.append({"epoch": epoch + 1, "loss": loss, "accuracy": metrics["accuracy"]})
    split_metrics: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, list[dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        logits = matrices[split_name]["X"] @ weights + bias
        probs = _softmax(logits)
        preds = np.argmax(probs, axis=1)
        split_metrics[split_name] = _compute_metrics(matrices[split_name]["y"], preds)
        predictions[split_name] = [
            {"sample_id": sample_id, "prediction": int(pred)}
            for sample_id, pred in zip(matrices[split_name]["sample_ids"], preds)
        ]
    importance = {
        spec.feature_node_id: float(np.linalg.norm(weights[idx]))
        for idx, spec in enumerate(plan.feature_specs)
    }
    return {
        "model_build_plan": plan.model_dump(),
        "runtime_backend": "numpy_fallback",
        "training_curves": curves,
        "checkpoint": {"weights": weights.tolist(), "bias": bias.tolist()},
        "metrics": split_metrics,
        "predictions": predictions,
        "importance": importance,
    }
