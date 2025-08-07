from __future__ import annotations

import base64
import io
from typing import Dict, Any

import joblib
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC


def _build_estimator(algorithm: str) -> Any:
    """Return a scikit-learn estimator based on ``algorithm``."""
    if algorithm.upper() == "SVM":
        return SVC(probability=True, random_state=42)
    return RandomForestClassifier(random_state=42)


def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train a model per dataset, evaluate, and perform ensemble inference.
    """
    if not datasets or pd is None:
        return {
            "models": {},
            "ensemble_metrics": {"accuracy": 0.0, "f1": 0.0},
            "metrics_markdown": "Pandas not available or no datasets provided.",
        }

    models: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, np.ndarray] = {}
    probas: Dict[str, np.ndarray] = {}
    y_truth = None  # Ground truth for consistent ensemble evaluation

    for node_id, data in datasets.items():
        X_train, y_train = data.get("X_train"), data.get("y_train")
        X_test, y_test = data.get("X_test"), data.get("y_test")

        if y_truth is None and isinstance(y_test, np.ndarray) and y_test.size > 0:
            y_truth = y_test

        if not all(isinstance(arr, np.ndarray) for arr in [X_train, y_train, X_test, y_test]):
            continue

        est = _build_estimator(algorithm)

        # --- Cross-validation ---
        cv_acc, cv_f1, cv_acc_std, cv_f1_std = 0.0, 0.0, 0.0, 0.0
        if cv_folds >= 2 and len(np.unique(y_train)) > 1:
            cv = min(cv_folds, len(np.unique(y_train)))
            if cv >= 2:
                scoring = {
                    'accuracy': 'accuracy',
                    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
                }
                cv_scores = cross_validate(est, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                cv_acc = np.mean(cv_scores['test_accuracy'])
                cv_f1 = np.mean(cv_scores['test_f1_weighted'])
                cv_acc_std = np.std(cv_scores['test_accuracy'])
                cv_f1_std = np.std(cv_scores['test_f1_weighted'])

        # --- Final model training and evaluation ---
        acc, f1 = 0.0, 0.0
        if len(np.unique(y_train)) > 1:
            est.fit(X_train, y_train)
            if y_test.size > 0 and X_test.shape[0] == y_test.shape[0]:
                y_pred = est.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                
                # FIX: Store predictions and probabilities in dictionaries with node_id as key
                predictions[node_id] = y_pred
                if hasattr(est, "predict_proba"):
                    probas[node_id] = est.predict_proba(X_test)
        
        buf = io.BytesIO()
        joblib.dump(est, buf)
        model_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        models[node_id] = {
            "metrics": {
                "accuracy": acc,
                "f1": f1,
                "cv_accuracy": cv_acc,
                "cv_f1": cv_f1,
                "cv_accuracy_std": cv_acc_std,
                "cv_f1_std": cv_f1_std,
            },
            "model_b64": model_b64,
        }

    # --- Ensemble Inference ---
    ensemble_metrics = {"accuracy": 0.0, "f1": 0.0}
    if y_truth is not None and predictions:
        high_quality_node_ids = [
            node_id for node_id, model_data in models.items()
            if model_data["metrics"].get("cv_accuracy", 0) > 0.90
        ]
        
        print(f"--- Ensemble: Found {len(high_quality_node_ids)} models with CV accuracy > 90% for ensembling.")
        if high_quality_node_ids:
            print(f"--- High-quality models: {high_quality_node_ids}")

            if ensemble_method == "soft_voting" and probas:
                valid_probas = [
                    probas[node_id] for node_id in high_quality_node_ids
                    if node_id in probas and probas[node_id].shape[0] == len(y_truth)
                ]
                if valid_probas:
                    avg_proba = np.mean(np.stack(valid_probas), axis=0)
                    ensemble_pred = np.argmax(avg_proba, axis=1)
            else:
                valid_preds = [
                    predictions[node_id] for node_id in high_quality_node_ids
                    if node_id in predictions and len(predictions[node_id]) == len(y_truth)
                ]
                if valid_preds:
                    preds_arr = np.stack(valid_preds)
                    ensemble_pred = mode(preds_arr, axis=0, keepdims=False).mode
            
            if 'ensemble_pred' in locals():
                ensemble_metrics["accuracy"] = accuracy_score(y_truth, ensemble_pred)
                ensemble_metrics["f1"] = f1_score(y_truth, ensemble_pred, average="weighted", zero_division=0)

    # --- Generate Markdown Report ---
    metrics_data = {node_id: m["metrics"] for node_id, m in models.items()}
    metrics_df = pd.DataFrame.from_dict(metrics_data, orient="index")
    
    if ensemble_metrics["accuracy"] > 0:
        for col in metrics_df.columns:
            if col not in ensemble_metrics:
                ensemble_metrics[col] = np.nan
        ensemble_df = pd.DataFrame([ensemble_metrics], index=["ensemble"])
        metrics_df = pd.concat([metrics_df, ensemble_df])

    return {
        "models": models,
        "ensemble_metrics": ensemble_metrics,
        "metrics_markdown": metrics_df.to_markdown(floatfmt=".6f"),
    }


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    dummy = {
        "fft_01": {
            "X_train": rng.random((20, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_test": rng.integers(0, 2, 10),
        },
        "psd_02": {
            "X_train": rng.random((20, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_test": rng.integers(0, 2, 10),
        },
    }
    print({"before": list(dummy.keys())})
    out = shallow_ml_agent(dummy)
    print({"after": out["ensemble_metrics"]})

