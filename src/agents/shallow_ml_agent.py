from __future__ import annotations

import base64
import io
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BALANCED_MODEL_POOL = [
    "RandomForest",
    "ExtraTrees",
    "SVM",
    "LogReg",
    "KNN",
    "MLP",
]

MODEL_COMPLEXITY_ORDER = {
    "LogReg": 0,
    "KNN": 1,
    "SVM": 2,
    "RandomForest": 3,
    "ExtraTrees": 4,
    "MLP": 5,
}


def _build_estimator(algorithm: str) -> Any:
    normalized = str(algorithm or "").strip()
    upper = normalized.upper()
    if upper == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(probability=True, class_weight="balanced", random_state=42),
        )
    if upper in {"LOGREG", "LOGISTICREGRESSION"}:
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
    if upper == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
        )
    if upper == "EXTRATREES":
        return ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    if upper == "MLP":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                early_stopping=True,
                max_iter=200,
                random_state=42,
            ),
        )
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )


def _normalize_candidate_algorithms(
    algorithm: str,
    candidate_algorithms: List[str] | None,
) -> List[str]:
    if candidate_algorithms:
        return [str(item) for item in candidate_algorithms]
    normalized = str(algorithm or "").strip()
    if normalized.lower() == "balanced_pool":
        return list(BALANCED_MODEL_POOL)
    return [normalized or "RandomForest"]


def _as_2d_matrix(array: Any) -> tuple[np.ndarray | None, int]:
    if not isinstance(array, np.ndarray):
        return None, 0
    if array.size == 0:
        n_samples = int(array.shape[0]) if array.ndim >= 1 else 0
        return np.empty((n_samples, 0), dtype=float), 0
    if array.ndim == 1:
        return np.asarray(array, dtype=float).reshape(-1, 1), 1
    feature_dim = int(np.prod(array.shape[1:])) if array.ndim > 1 else 1
    return np.asarray(array, dtype=float).reshape(array.shape[0], -1), feature_dim


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _selection_score(metrics: Dict[str, float]) -> float:
    for key in ("val_macro_f1", "val_accuracy", "macro_f1"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not np.isnan(value):
            return float(value)
    return 0.0


def _softmax(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    arr = arr - np.max(arr)
    weights = np.exp(arr)
    total = float(np.sum(weights))
    if total <= 0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=float)
    return weights / total


def _vote_from_predictions(predictions: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not predictions:
        return np.asarray([], dtype=float)
    valid = [np.asarray(pred) for pred in predictions]
    labels = np.unique(np.concatenate(valid))
    score_matrix = np.zeros((valid[0].shape[0], labels.size), dtype=float)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for pred, weight in zip(valid, weights):
        for row_index, label in enumerate(pred):
            score_matrix[row_index, label_to_index[label]] += float(weight)
    return labels[np.argmax(score_matrix, axis=1)]


def _probability_average(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray | None:
    if not probas:
        return None
    shapes = {tuple(proba.shape) for proba in probas}
    if len(shapes) != 1:
        return None
    stack = np.stack(probas)
    return np.average(stack, axis=0, weights=weights)


def _as_object_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=object)
    if isinstance(values, np.ndarray):
        return values
    return np.asarray(list(values), dtype=object)


def _serialize_estimator(estimator: Any) -> str:
    buffer = io.BytesIO()
    try:
        joblib.dump(estimator, buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _fit_and_score_candidate(
    node_id: str,
    data: Dict[str, Any],
    *,
    algorithm: str,
    cv_folds: int,
) -> Dict[str, Any]:
    X_train_raw, y_train = data.get("X_train"), data.get("y_train")
    X_val_raw, y_val = data.get("X_val"), data.get("y_val")
    X_test_raw, y_test = data.get("X_test"), data.get("y_test")

    X_train, feature_dim = _as_2d_matrix(X_train_raw)
    X_val, _ = _as_2d_matrix(X_val_raw) if isinstance(X_val_raw, np.ndarray) else (None, 0)
    X_test, _ = _as_2d_matrix(X_test_raw)
    if not isinstance(y_train, np.ndarray) or not isinstance(y_test, np.ndarray):
        return {
            "node_id": node_id,
            "algorithm": algorithm,
            "can_fit": False,
            "feature_dim": 0,
            "failure_reason": "missing_numpy_splits",
        }
    if X_train is None or X_test is None:
        return {
            "node_id": node_id,
            "algorithm": algorithm,
            "can_fit": False,
            "feature_dim": 0,
            "failure_reason": "invalid_feature_matrix",
        }

    estimator = _build_estimator(algorithm)
    cv_acc = cv_f1 = cv_macro_f1 = 0.0
    cv_acc_std = cv_f1_std = cv_macro_f1_std = 0.0
    can_fit = len(np.unique(y_train)) > 1 and X_train.shape[0] > 0 and X_train.shape[0] == len(y_train) and feature_dim > 0

    if can_fit and cv_folds >= 2:
        _, class_counts = np.unique(y_train, return_counts=True)
        min_class_count = int(class_counts.min()) if class_counts.size else 0
        cv = min(int(cv_folds), min_class_count)
        if cv >= 2:
            scoring = {
                "accuracy": "accuracy",
                "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
                "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            }
            cv_scores = cross_validate(estimator, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_acc = float(np.mean(cv_scores["test_accuracy"]))
            cv_f1 = float(np.mean(cv_scores["test_f1_weighted"]))
            cv_macro_f1 = float(np.mean(cv_scores["test_f1_macro"]))
            cv_acc_std = float(np.std(cv_scores["test_accuracy"]))
            cv_f1_std = float(np.std(cv_scores["test_f1_weighted"]))
            cv_macro_f1_std = float(np.std(cv_scores["test_f1_macro"]))

    train_metrics = {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0}
    val_metrics = {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0}
    test_metrics = {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0}
    train_pred = val_pred = test_pred = None
    train_proba = val_proba = test_proba = None
    model_b64 = ""

    if can_fit:
        estimator.fit(X_train, y_train)
        if X_train.shape[0] == len(y_train):
            train_pred = estimator.predict(X_train)
            train_metrics = _metric_bundle(np.asarray(y_train), np.asarray(train_pred))
        if X_val is not None and isinstance(y_val, np.ndarray) and X_val.shape[0] == len(y_val) and X_val.shape[0] > 0:
            val_pred = estimator.predict(X_val)
            val_metrics = _metric_bundle(np.asarray(y_val), np.asarray(val_pred))
        if X_test.shape[0] == len(y_test) and X_test.shape[0] > 0:
            test_pred = estimator.predict(X_test)
            test_metrics = _metric_bundle(np.asarray(y_test), np.asarray(test_pred))
            if hasattr(estimator, "predict_proba"):
                test_proba = estimator.predict_proba(X_test)
            if X_val is not None and isinstance(y_val, np.ndarray) and X_val.shape[0] == len(y_val) and X_val.shape[0] > 0 and hasattr(estimator, "predict_proba"):
                val_proba = estimator.predict_proba(X_val)
            if hasattr(estimator, "predict_proba") and X_train.shape[0] == len(y_train) and X_train.shape[0] > 0:
                train_proba = estimator.predict_proba(X_train)
        model_b64 = _serialize_estimator(estimator)

    selection_score = _selection_score(
        {
            "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
        }
    )
    return {
        "node_id": node_id,
        "algorithm": algorithm,
        "feature_dim": int(feature_dim),
        "can_fit": bool(can_fit),
        "failure_reason": "" if can_fit else "insufficient_training_signal",
        "metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "f1": float(test_metrics["f1"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_f1": float(train_metrics["f1"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_f1": float(val_metrics["f1"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_f1": float(test_metrics["f1"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "cv_accuracy": float(cv_acc),
            "cv_f1": float(cv_f1),
            "cv_macro_f1": float(cv_macro_f1),
            "cv_accuracy_std": float(cv_acc_std),
            "cv_f1_std": float(cv_f1_std),
            "cv_macro_f1_std": float(cv_macro_f1_std),
            "feature_dim": int(feature_dim),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]) if X_val is not None else 0,
            "test_samples": int(X_test.shape[0]),
            "selection_score": float(selection_score),
            "algorithm": algorithm,
            "failure_reason": "" if can_fit else "insufficient_training_signal",
        },
        "model_b64": model_b64,
        "train_pred": np.asarray(train_pred) if train_pred is not None else np.asarray([], dtype=int),
        "val_pred": np.asarray(val_pred) if val_pred is not None else np.asarray([], dtype=int),
        "test_pred": np.asarray(test_pred) if test_pred is not None else np.asarray([], dtype=int),
        "train_proba": np.asarray(train_proba) if train_proba is not None else None,
        "val_proba": np.asarray(val_proba) if val_proba is not None else None,
        "test_proba": np.asarray(test_proba) if test_proba is not None else None,
        "val_window_ids": _as_object_array(data.get("val_window_ids")),
        "test_window_ids": _as_object_array(data.get("test_window_ids")),
        "y_val": np.asarray(y_val) if isinstance(y_val, np.ndarray) else np.asarray([], dtype=int),
        "y_test": np.asarray(y_test) if isinstance(y_test, np.ndarray) else np.asarray([], dtype=int),
    }


def _parallel_workers_default() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
    candidate_algorithms: List[str] | None = None,
    parallel_workers: int | None = None,
) -> Dict[str, Any]:
    """Train one or more shallow models per leaf and produce node-level and ensemble metrics."""
    if not datasets or pd is None:
        return {
            "models": {},
            "node_level_results": [],
            "ensemble_metrics": {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0, "ensemble_macro_f1": 0.0},
            "weighted_ensemble_metrics": {"accuracy": 0.0, "f1": 0.0, "macro_f1": 0.0, "ensemble_macro_f1": 0.0},
            "final_selection": {},
            "metrics_markdown": "Pandas not available or no datasets provided.",
        }

    selected_algorithms = _normalize_candidate_algorithms(algorithm, candidate_algorithms)
    tasks = [
        (node_id, data, algorithm_name)
        for node_id, data in datasets.items()
        for algorithm_name in selected_algorithms
    ]
    if parallel_workers is not None:
        worker_count = max(1, int(parallel_workers))
    else:
        worker_count = 1

    if worker_count == 1:
        candidate_results = [
            _fit_and_score_candidate(node_id, data, algorithm=algorithm_name, cv_folds=cv_folds)
            for node_id, data, algorithm_name in tasks
        ]
    else:
        candidate_results = Parallel(n_jobs=worker_count, backend="loky")(
            delayed(_fit_and_score_candidate)(node_id, data, algorithm=algorithm_name, cv_folds=cv_folds)
            for node_id, data, algorithm_name in tasks
        )

    grouped_results: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for result in candidate_results:
        grouped_results[str(result["node_id"])].append(result)

    models: Dict[str, Dict[str, Any]] = {}
    node_level_results: list[Dict[str, Any]] = []
    test_predictions: Dict[str, np.ndarray] = {}
    test_probas: Dict[str, np.ndarray] = {}
    val_predictions: Dict[str, np.ndarray] = {}
    val_probas: Dict[str, np.ndarray] = {}
    selection_scores: Dict[str, float] = {}
    y_truth_test = None
    y_truth_val = None
    val_window_ids: Dict[str, np.ndarray] = {}
    test_window_ids: Dict[str, np.ndarray] = {}

    for node_id, data in datasets.items():
        y_val = data.get("y_val")
        y_test = data.get("y_test")
        if y_truth_test is None and isinstance(y_test, np.ndarray) and y_test.size > 0:
            y_truth_test = np.asarray(y_test)
        if y_truth_val is None and isinstance(y_val, np.ndarray) and y_val.size > 0:
            y_truth_val = np.asarray(y_val)

        candidates = grouped_results.get(node_id, [])
        if not candidates:
            continue
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float((item.get("metrics") or {}).get("val_macro_f1", 0.0)),
                -float((item.get("metrics") or {}).get("val_accuracy", 0.0)),
                MODEL_COMPLEXITY_ORDER.get(str(item.get("algorithm")), 999),
                str(item.get("algorithm", "")),
            ),
        )
        best = ranked[0]
        best_metrics = dict(best.get("metrics") or {})
        best_metrics["candidate_algorithms"] = list(selected_algorithms)
        models[node_id] = {
            "metrics": dict(best_metrics),
            "model_b64": str(best.get("model_b64", "")),
            "selected_algorithm": str(best.get("algorithm", "")),
            "candidate_algorithms": list(selected_algorithms),
        }

        row = {
            "node_id": node_id,
            "leaf_id": str(data.get("leaf_id", node_id)),
            "branch_id": str(data.get("branch_id", node_id)),
            "branch_summary": str(data.get("branch_summary", node_id)),
            **best_metrics,
            "selected_single_best": False,
            "selected_in_ensemble": False,
            "failure_reason": str(best.get("failure_reason", "")),
        }
        node_level_results.append(row)

        if best.get("test_pred") is not None and np.asarray(best.get("test_pred")).size > 0:
            test_predictions[node_id] = np.asarray(best.get("test_pred"))
        if best.get("test_proba") is not None:
            test_probas[node_id] = np.asarray(best.get("test_proba"))
        if best.get("val_pred") is not None and np.asarray(best.get("val_pred")).size > 0:
            val_predictions[node_id] = np.asarray(best.get("val_pred"))
        if best.get("val_proba") is not None:
            val_probas[node_id] = np.asarray(best.get("val_proba"))
        if best.get("val_window_ids") is not None and np.asarray(best.get("val_window_ids")).size > 0:
            val_window_ids[node_id] = _as_object_array(best.get("val_window_ids"))
        if best.get("test_window_ids") is not None and np.asarray(best.get("test_window_ids")).size > 0:
            test_window_ids[node_id] = _as_object_array(best.get("test_window_ids"))
        selection_scores[node_id] = float(best_metrics.get("selection_score", 0.0))

    best_single_leaf = None
    best_single_metrics: Dict[str, Any] = {}
    if node_level_results:
        ranked_rows = sorted(
            node_level_results,
            key=lambda row: (
                -float(row.get("val_macro_f1", 0.0)),
                -float(row.get("val_accuracy", 0.0)),
                MODEL_COMPLEXITY_ORDER.get(str(row.get("algorithm", "")), 999),
                int(row.get("feature_dim", 0)),
                str(row.get("node_id", "")),
            ),
        )
        best_single_metrics = dict(ranked_rows[0])
        best_single_leaf = str(ranked_rows[0]["node_id"])
        best_single_metrics["selected_single_best"] = True
        if best_single_leaf in models:
            models[best_single_leaf]["metrics"]["selected_single_best"] = True
        for row in node_level_results:
            row["selected_single_best"] = row["node_id"] == best_single_leaf

    valid_node_ids = [node_id for node_id in models if node_id in test_predictions]
    valid_weights = np.asarray([selection_scores.get(node_id, 0.0) for node_id in valid_node_ids], dtype=float)
    if valid_node_ids:
        if np.allclose(valid_weights.sum(), 0.0):
            valid_weights = np.ones_like(valid_weights, dtype=float)
        valid_weights = _softmax(valid_weights)

    weighted_ensemble_metrics = {
        "accuracy": 0.0,
        "f1": 0.0,
        "macro_f1": 0.0,
        "ensemble_macro_f1": 0.0,
        "val_accuracy": 0.0,
        "val_f1": 0.0,
        "val_macro_f1": 0.0,
        "test_accuracy": 0.0,
        "test_f1": 0.0,
        "test_macro_f1": 0.0,
    }
    weighted_val_pred: np.ndarray | None = None
    weighted_test_pred: np.ndarray | None = None
    weighted_val_proba: np.ndarray | None = None
    weighted_test_proba: np.ndarray | None = None

    if valid_node_ids:
        if ensemble_method == "soft_voting":
            test_proba_nodes = [node_id for node_id in valid_node_ids if node_id in test_probas]
            test_proba_weights = np.asarray([selection_scores.get(node_id, 0.0) for node_id in test_proba_nodes], dtype=float)
            if test_proba_nodes:
                if np.allclose(test_proba_weights.sum(), 0.0):
                    test_proba_weights = np.ones_like(test_proba_weights, dtype=float)
                weighted_test_proba = _probability_average([test_probas[nid] for nid in test_proba_nodes], _softmax(test_proba_weights))
            val_proba_nodes = [node_id for node_id in valid_node_ids if node_id in val_probas]
            val_proba_weights = np.asarray([selection_scores.get(node_id, 0.0) for node_id in val_proba_nodes], dtype=float)
            if val_proba_nodes:
                if np.allclose(val_proba_weights.sum(), 0.0):
                    val_proba_weights = np.ones_like(val_proba_weights, dtype=float)
                weighted_val_proba = _probability_average([val_probas[nid] for nid in val_proba_nodes], _softmax(val_proba_weights))
            if weighted_test_proba is not None and y_truth_test is not None and weighted_test_proba.shape[0] == len(y_truth_test):
                weighted_test_pred = np.argmax(weighted_test_proba, axis=1)
                weighted_ensemble_metrics.update(
                    {
                        **_metric_bundle(y_truth_test, weighted_test_pred),
                        "ensemble_macro_f1": float(f1_score(y_truth_test, weighted_test_pred, average="macro", zero_division=0)),
                        "test_accuracy": float(accuracy_score(y_truth_test, weighted_test_pred)),
                        "test_f1": float(f1_score(y_truth_test, weighted_test_pred, average="weighted", zero_division=0)),
                        "test_macro_f1": float(f1_score(y_truth_test, weighted_test_pred, average="macro", zero_division=0)),
                    }
                )
            if weighted_val_proba is not None and y_truth_val is not None and weighted_val_proba.shape[0] == len(y_truth_val):
                weighted_val_pred = np.argmax(weighted_val_proba, axis=1)
                weighted_ensemble_metrics.update(
                    {
                        "val_accuracy": float(accuracy_score(y_truth_val, weighted_val_pred)),
                        "val_f1": float(f1_score(y_truth_val, weighted_val_pred, average="weighted", zero_division=0)),
                        "val_macro_f1": float(f1_score(y_truth_val, weighted_val_pred, average="macro", zero_division=0)),
                    }
                )
        else:
            test_truth_len = len(y_truth_test) if y_truth_test is not None else 0
            valid_test_preds = [
                test_predictions[nid]
                for nid in valid_node_ids
                if nid in test_predictions and len(test_predictions[nid]) == test_truth_len
            ]
            if valid_test_preds and y_truth_test is not None:
                weighted_test_pred = _vote_from_predictions(valid_test_preds, valid_weights)
                weighted_ensemble_metrics.update(
                    {
                        **_metric_bundle(y_truth_test, weighted_test_pred),
                        "ensemble_macro_f1": float(f1_score(y_truth_test, weighted_test_pred, average="macro", zero_division=0)),
                        "test_accuracy": float(accuracy_score(y_truth_test, weighted_test_pred)),
                        "test_f1": float(f1_score(y_truth_test, weighted_test_pred, average="weighted", zero_division=0)),
                        "test_macro_f1": float(f1_score(y_truth_test, weighted_test_pred, average="macro", zero_division=0)),
                    }
                )
            val_truth_len = len(y_truth_val) if y_truth_val is not None else 0
            valid_val_preds = [
                val_predictions[nid]
                for nid in valid_node_ids
                if nid in val_predictions and y_truth_val is not None and len(val_predictions[nid]) == val_truth_len
            ]
            if valid_val_preds and y_truth_val is not None:
                weighted_val_pred = _vote_from_predictions(valid_val_preds, valid_weights)
                weighted_ensemble_metrics.update(
                    {
                        "val_accuracy": float(accuracy_score(y_truth_val, weighted_val_pred)),
                        "val_f1": float(f1_score(y_truth_val, weighted_val_pred, average="weighted", zero_division=0)),
                        "val_macro_f1": float(f1_score(y_truth_val, weighted_val_pred, average="macro", zero_division=0)),
                    }
                )

        for node_id in valid_node_ids:
            if node_id in models:
                models[node_id]["metrics"]["selected_in_ensemble"] = True
        for row in node_level_results:
            row["selected_in_ensemble"] = row["node_id"] in valid_node_ids
        if best_single_leaf:
            best_single_metrics["selected_in_ensemble"] = best_single_leaf in valid_node_ids

    weighted_ensemble_metrics["ensemble_macro_f1"] = float(weighted_ensemble_metrics.get("macro_f1", 0.0))
    ensemble_metrics = dict(weighted_ensemble_metrics)

    final_selection: Dict[str, Any] = {"selection_basis": "val_macro_f1"}
    if best_single_leaf:
        final_selection["best_single_leaf"] = best_single_leaf
        final_selection["best_single_leaf_metrics"] = best_single_metrics
    if valid_node_ids:
        final_selection["weighted_ensemble"] = weighted_ensemble_metrics
    best_single_score = float(best_single_metrics.get("val_macro_f1", 0.0)) if best_single_metrics else 0.0
    ensemble_score = float(weighted_ensemble_metrics.get("val_macro_f1", 0.0))
    if best_single_leaf and best_single_score >= ensemble_score:
        final_selection["final_choice"] = "best_single_leaf"
    elif valid_node_ids:
        final_selection["final_choice"] = "weighted_ensemble"
    else:
        final_selection["final_choice"] = "none"

    reference_window_ids_val = _as_object_array(next(iter(val_window_ids.values()))) if val_window_ids else np.asarray([], dtype=object)
    reference_window_ids_test = _as_object_array(next(iter(test_window_ids.values()))) if test_window_ids else np.asarray([], dtype=object)
    best_single_payload: Dict[str, Any] = {}
    if best_single_leaf:
        best_single_payload = {
            "leaf_id": best_single_leaf,
            "algorithm": str(best_single_metrics.get("algorithm", "")),
            "val_pred": np.asarray(val_predictions.get(best_single_leaf, np.asarray([], dtype=int))),
            "test_pred": np.asarray(test_predictions.get(best_single_leaf, np.asarray([], dtype=int))),
            "val_proba": np.asarray(val_probas[best_single_leaf]) if best_single_leaf in val_probas else None,
            "test_proba": np.asarray(test_probas[best_single_leaf]) if best_single_leaf in test_probas else None,
            "val_macro_f1": float(best_single_metrics.get("val_macro_f1", 0.0)),
        }
    weighted_payload: Dict[str, Any] = {}
    if valid_node_ids:
        weighted_payload = {
            "member_node_ids": list(valid_node_ids),
            "weights": np.asarray(valid_weights, dtype=float),
            "val_pred": np.asarray(weighted_val_pred) if weighted_val_pred is not None else np.asarray([], dtype=int),
            "test_pred": np.asarray(weighted_test_pred) if weighted_test_pred is not None else np.asarray([], dtype=int),
            "val_proba": np.asarray(weighted_val_proba) if weighted_val_proba is not None else None,
            "test_proba": np.asarray(weighted_test_proba) if weighted_test_proba is not None else None,
            "val_macro_f1": float(weighted_ensemble_metrics.get("val_macro_f1", 0.0)),
        }
    final_choice_strategy = str(final_selection.get("final_choice") or "none")
    final_choice_payload = {
        "strategy": final_choice_strategy,
        "val_macro_f1": float(best_single_metrics.get("val_macro_f1", 0.0))
        if final_choice_strategy == "best_single_leaf"
        else float(weighted_ensemble_metrics.get("val_macro_f1", 0.0)),
        "val_pred": np.asarray(best_single_payload.get("val_pred", np.asarray([], dtype=int)))
        if final_choice_strategy == "best_single_leaf"
        else np.asarray(weighted_payload.get("val_pred", np.asarray([], dtype=int))),
        "test_pred": np.asarray(best_single_payload.get("test_pred", np.asarray([], dtype=int)))
        if final_choice_strategy == "best_single_leaf"
        else np.asarray(weighted_payload.get("test_pred", np.asarray([], dtype=int))),
        "val_proba": best_single_payload.get("val_proba")
        if final_choice_strategy == "best_single_leaf"
        else weighted_payload.get("val_proba"),
        "test_proba": best_single_payload.get("test_proba")
        if final_choice_strategy == "best_single_leaf"
        else weighted_payload.get("test_proba"),
        "algorithm": str(best_single_metrics.get("algorithm", "")) if final_choice_strategy == "best_single_leaf" else "weighted_ensemble",
    }
    if final_choice_strategy == "best_single_leaf":
        reference_window_ids_val = _as_object_array(val_window_ids.get(best_single_leaf, reference_window_ids_val))
        reference_window_ids_test = _as_object_array(test_window_ids.get(best_single_leaf, reference_window_ids_test))
    selection_predictions = {
        "selection_basis": "val_macro_f1",
        "window_ids_val": reference_window_ids_val,
        "window_ids_test": reference_window_ids_test,
        "y_val": np.asarray(y_truth_val) if y_truth_val is not None else np.asarray([], dtype=int),
        "y_test": np.asarray(y_truth_test) if y_truth_test is not None else np.asarray([], dtype=int),
        "best_single_leaf": best_single_payload,
        "weighted_ensemble": weighted_payload,
        "final_choice": final_choice_payload,
        "final_choice_strategy": final_choice_strategy,
    }

    metrics_df = pd.DataFrame(node_level_results)
    summary_rows: list[Dict[str, Any]] = []
    if best_single_leaf and best_single_metrics:
        summary_rows.append(
            {
                **best_single_metrics,
                "node_id": best_single_leaf,
                "result_type": "best_single_leaf",
            }
        )
    if valid_node_ids:
        summary_rows.append(
            {
                "node_id": "weighted_ensemble",
                "result_type": "weighted_ensemble",
                "algorithm": "weighted_ensemble",
                **weighted_ensemble_metrics,
            }
        )
    if summary_rows:
        metrics_df = pd.concat([metrics_df, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)

    try:
        metrics_markdown = metrics_df.to_markdown(floatfmt=".6f", index=False)
    except ImportError:
        metrics_markdown = metrics_df.to_string(index=False, float_format=lambda value: f"{value:.6f}")

    return {
        "models": models,
        "node_level_results": node_level_results,
        "ensemble_metrics": ensemble_metrics,
        "weighted_ensemble_metrics": weighted_ensemble_metrics,
        "final_selection": final_selection,
        "selection_predictions": selection_predictions,
        "metrics_markdown": metrics_markdown,
        "candidate_algorithms": list(selected_algorithms),
        "parallel_workers": int(worker_count),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dummy = {
        "fft_01": {
            "X_train": rng.random((20, 5)),
            "X_val": rng.random((10, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_val": rng.integers(0, 2, 10),
            "y_test": rng.integers(0, 2, 10),
        },
        "psd_02": {
            "X_train": rng.random((20, 5)),
            "X_val": rng.random((10, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_val": rng.integers(0, 2, 10),
            "y_test": rng.integers(0, 2, 10),
        },
    }
    out = shallow_ml_agent(dummy, algorithm="balanced_pool", parallel_workers=2)
    print({"final_selection": out["final_selection"], "candidate_algorithms": out["candidate_algorithms"]})
