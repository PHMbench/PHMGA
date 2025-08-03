from __future__ import annotations

import base64
from typing import Dict, Any

import joblib
import numpy as np
import io
try:  # pragma: no cover - pandas might not be installed
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from scipy.stats import mode


def _build_estimator(algorithm: str) -> Any:
    """Return a scikit-learn estimator based on ``algorithm``."""
    if algorithm.upper() == "SVM":
        # probability=True to enable soft voting via predict_proba
        return SVC(probability=True)
    # Default algorithm
    return RandomForestClassifier()


def shallow_ml_agent(
    datasets: Dict[str, Dict[str, Any]],
    *,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Train a model per dataset and perform ensemble inference.

    Parameters
    ----------
    datasets : Dict[str, Dict[str, Any]]
        Mapping from node id to dataset splits.
    algorithm : str, optional
        Base estimator to train (``"RandomForest"`` or ``"SVM"``).
    ensemble_method : str, optional
        ``"hard_voting"`` or ``"soft_voting"``.
    cv_folds : int, optional
        Number of cross-validation folds. Defaults to 5.

    Returns
    -------
    Dict[str, Any]
        Model metrics, ensemble metrics and a markdown table summary.
    """

    if not datasets:
        return {
            "models": {},
            "ensemble_metrics": {"accuracy": 0.0, "f1": 0.0},
            "metrics_markdown": "| model | accuracy | f1 |\n|---|---|---|",
        }

    models: Dict[str, Any] = {}
    predictions = []
    probas = []
    y_truth = None

    for node_id, data in datasets.items():
        X_train = np.asarray(data.get("X_train", []))
        X_test = np.asarray(data.get("X_test", []))
        y_train = np.asarray(data.get("y_train", []))
        y_test = np.asarray(data.get("y_test", []))

        if y_truth is None and y_test.size:
            y_truth = y_test

        est = _build_estimator(algorithm)

        cv = min(cv_folds, len(y_train)) if len(y_train) else 0
        if cv >= 2 and len(np.unique(y_train)) > 1:
            acc_scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy")
            f1_scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="f1")
            cv_acc = float(acc_scores.mean())
            cv_f1 = float(f1_scores.mean())
        else:
            cv_acc = cv_f1 = 0.0

        if len(np.unique(y_train)) > 1:
            est.fit(X_train, y_train)
        else:
            est.fit(X_train, np.zeros(len(X_train)))

        y_pred = est.predict(X_test) if X_test.size else np.array([])
        y_proba = est.predict_proba(X_test) if hasattr(est, "predict_proba") and X_test.size else None

        acc = accuracy_score(y_test, y_pred) if y_test.size else 0.0
        f1 = f1_score(y_test, y_pred) if y_test.size else 0.0

        predictions.append(y_pred)
        probas.append(y_proba)

        buf = io.BytesIO()
        joblib.dump(est, buf)
        model_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        models[node_id] = {
            "metrics": {"accuracy": acc, "f1": f1, "cv_accuracy": cv_acc, "cv_f1": cv_f1},
            "model_b64": model_b64,
        }

    ensemble_metrics = {"accuracy": 0.0, "f1": 0.0}
    if y_truth is not None and predictions:
        preds_arr = np.stack(predictions)
        if ensemble_method == "soft_voting" and all(p is not None for p in probas):
            avg_proba = np.mean(np.stack(probas), axis=0)
            ensemble_pred = avg_proba.argmax(axis=1)
        else:
            ensemble_pred = mode(preds_arr, axis=0, keepdims=False).mode
        ensemble_metrics["accuracy"] = float(accuracy_score(y_truth, ensemble_pred))
        ensemble_metrics["f1"] = float(f1_score(y_truth, ensemble_pred))

    rows = []
    for node_id, info in models.items():
        rows.append({"model": node_id, **info["metrics"]})
    rows.append({"model": "ensemble", **ensemble_metrics})
    if pd is not None:
        metrics_markdown = pd.DataFrame(rows).to_markdown(index=False)
    else:  # fallback simple markdown
        headers = list({h for r in rows for h in r.keys()})
        lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
        metrics_markdown = "\n".join(lines)

    return {
        "models": models,
        "ensemble_metrics": ensemble_metrics,
        "metrics_markdown": metrics_markdown,
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

