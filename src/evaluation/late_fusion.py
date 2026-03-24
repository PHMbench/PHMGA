from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


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


def _metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _optional_array(value: Any, *, dtype: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _align_predictions(
    *,
    target_ids: np.ndarray,
    source_ids: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    index_map = {str(sample_id): index for index, sample_id in enumerate(source_ids.tolist())}
    pred_out = []
    proba_rows = [] if probabilities is not None else None
    for sample_id in target_ids.tolist():
        src_index = index_map[str(sample_id)]
        pred_out.append(predictions[src_index])
        if probabilities is not None and proba_rows is not None:
            proba_rows.append(probabilities[src_index])
    aligned_pred = np.asarray(pred_out)
    aligned_proba = np.asarray(proba_rows) if proba_rows is not None else None
    return aligned_pred, aligned_proba


def _weighted_vote(predictions: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not predictions:
        return np.asarray([], dtype=int)
    labels = np.unique(np.concatenate(predictions))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    score_matrix = np.zeros((predictions[0].shape[0], labels.size), dtype=float)
    for pred, weight in zip(predictions, weights):
        for row_index, label in enumerate(pred):
            score_matrix[row_index, label_to_index[label]] += float(weight)
    return labels[np.argmax(score_matrix, axis=1)]


def _weighted_probability_average(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray | None:
    if not probabilities:
        return None
    shapes = {tuple(proba.shape) for proba in probabilities}
    if len(shapes) != 1:
        return None
    stack = np.stack(probabilities)
    return np.average(stack, axis=0, weights=weights)


def load_run_prediction_payload(run_dir: str | Path) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json under {run_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection_predictions_path = Path(
        manifest.get("selection_predictions_path")
        or run_path / "selection_predictions.pkl"
    )
    payload = _load_pickle(selection_predictions_path)
    return {
        "run_dir": str(run_path),
        "provider": str(manifest.get("provider", "")),
        "model": str(manifest.get("model", "")),
        "paper_label": str(manifest.get("paper_label", "")) or str(manifest.get("model", "")),
        "run_type": str(manifest.get("run_type", "")),
        "selection_predictions": payload,
    }


def compute_cross_dag_late_fusion(run_dirs: Iterable[str | Path]) -> Dict[str, Any]:
    runs = [load_run_prediction_payload(run_dir) for run_dir in run_dirs]
    if not runs:
        raise ValueError("compute_cross_dag_late_fusion requires at least one run directory.")

    reference = runs[0]["selection_predictions"]
    ref_val_ids = _optional_array(reference.get("window_ids_val"), dtype=object)
    ref_test_ids = _optional_array(reference.get("window_ids_test"), dtype=object)
    y_val = _optional_array(reference.get("y_val"), dtype=int)
    y_test = _optional_array(reference.get("y_test"), dtype=int)
    if ref_val_ids.size == 0 or ref_test_ids.size == 0:
        raise ValueError("Missing validation/test window identifiers for late fusion.")

    aligned_val_preds: list[np.ndarray] = []
    aligned_test_preds: list[np.ndarray] = []
    aligned_val_probas: list[np.ndarray] = []
    aligned_test_probas: list[np.ndarray] = []
    weights_raw: list[float] = []
    run_rows: list[Dict[str, Any]] = []

    for run in runs:
        payload = run["selection_predictions"]
        final_choice = dict(payload.get("final_choice") or {})
        source_val_ids = _optional_array(payload.get("window_ids_val"), dtype=object)
        source_test_ids = _optional_array(payload.get("window_ids_test"), dtype=object)
        val_pred = _optional_array(final_choice.get("val_pred"), dtype=int)
        test_pred = _optional_array(final_choice.get("test_pred"), dtype=int)
        val_proba_raw = final_choice.get("val_proba")
        test_proba_raw = final_choice.get("test_proba")
        val_proba = np.asarray(val_proba_raw) if val_proba_raw is not None else None
        test_proba = np.asarray(test_proba_raw) if test_proba_raw is not None else None

        aligned_val_pred, aligned_val_proba = _align_predictions(
            target_ids=ref_val_ids,
            source_ids=source_val_ids,
            predictions=val_pred,
            probabilities=val_proba,
        )
        aligned_test_pred, aligned_test_proba = _align_predictions(
            target_ids=ref_test_ids,
            source_ids=source_test_ids,
            predictions=test_pred,
            probabilities=test_proba,
        )

        aligned_val_preds.append(aligned_val_pred)
        aligned_test_preds.append(aligned_test_pred)
        if aligned_val_proba is not None and aligned_test_proba is not None:
            aligned_val_probas.append(aligned_val_proba)
            aligned_test_probas.append(aligned_test_proba)

        weight_value = float(final_choice.get("val_macro_f1", 0.0) or 0.0)
        weights_raw.append(weight_value)
        run_rows.append(
            {
                "paper_label": run.get("paper_label", ""),
                "provider": run.get("provider", ""),
                "model": run.get("model", ""),
                "run_type": run.get("run_type", ""),
                "final_choice_strategy": str(payload.get("final_choice_strategy", "")),
                "selection_basis": str(payload.get("selection_basis", "")),
                "val_macro_f1_weight": weight_value,
            }
        )

    weights = _softmax(weights_raw)
    use_probability_average = len(aligned_val_probas) == len(runs) and len(aligned_test_probas) == len(runs)

    if use_probability_average:
        val_proba = _weighted_probability_average(aligned_val_probas, weights)
        test_proba = _weighted_probability_average(aligned_test_probas, weights)
        if val_proba is None or test_proba is None:
            use_probability_average = False
        else:
            val_pred = np.argmax(val_proba, axis=1)
            test_pred = np.argmax(test_proba, axis=1)
    if not use_probability_average:
        val_proba = None
        test_proba = None
        val_pred = _weighted_vote(aligned_val_preds, weights)
        test_pred = _weighted_vote(aligned_test_preds, weights)

    val_metrics = _metric_bundle(y_val, val_pred)
    test_metrics = _metric_bundle(y_test, test_pred)
    return {
        "selection_basis": "val_macro_f1",
        "fusion_method": "weighted_probability_average" if use_probability_average else "weighted_vote",
        "weights": [
            {
                **row,
                "softmax_weight": float(weight),
            }
            for row, weight in zip(run_rows, weights)
        ],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_val_windows": int(len(y_val)),
        "n_test_windows": int(len(y_test)),
    }
