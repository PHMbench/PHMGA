from __future__ import annotations

import numpy as np

from src.agents import shallow_ml_agent as shallow_module
from src.agents.shallow_ml_agent import shallow_ml_agent


def test_shallow_ml_agent_skips_cv_when_class_counts_are_too_small():
    datasets = {
        "fft_01_ch1": {
            "X_train": np.array([[0.0], [1.0], [2.0]]),
            "X_test": np.array([[0.1], [1.1], [2.1]]),
            "y_train": np.array([0, 1, 2]),
            "y_test": np.array([0, 1, 2]),
        }
    }

    result = shallow_ml_agent(datasets, cv_folds=5)

    metrics = result["models"]["fft_01_ch1"]["metrics"]
    assert metrics["cv_accuracy"] == 0.0
    assert metrics["cv_f1"] == 0.0


def test_shallow_ml_agent_reports_split_metrics_feature_dim_and_final_selection(monkeypatch):
    class DummyEstimator:
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.asarray([0 if row[0] < 0.5 else 1 for row in X])

        def predict_proba(self, X):
            pred = self.predict(X)
            proba = np.zeros((len(pred), 2), dtype=float)
            for idx, label in enumerate(pred):
                proba[idx, int(label)] = 1.0
            return proba

    monkeypatch.setattr(shallow_module, "_build_estimator", lambda algorithm: DummyEstimator())

    datasets = {
        "leaf_a": {
            "X_train": np.array([[0.0, 0.1], [0.1, 0.0], [0.9, 1.0], [1.0, 0.9]]),
            "X_val": np.array([[0.05, 0.0], [0.95, 1.0]]),
            "X_test": np.array([[0.02, 0.0], [0.98, 1.0]]),
            "y_train": np.array([0, 0, 1, 1]),
            "y_val": np.array([0, 1]),
            "y_test": np.array([0, 1]),
        },
        "leaf_b": {
            "X_train": np.array([[0.0, 0.2], [0.1, 0.1], [0.8, 0.9], [0.9, 0.8]]),
            "X_val": np.array([[0.75, 0.8], [0.85, 0.0]]),
            "X_test": np.array([[0.03, 0.05], [0.97, 0.95]]),
            "y_train": np.array([0, 0, 1, 1]),
            "y_val": np.array([1, 0]),
            "y_test": np.array([0, 1]),
        },
    }

    result = shallow_ml_agent(datasets, cv_folds=0, ensemble_method="hard_voting")

    leaf_a_metrics = result["models"]["leaf_a"]["metrics"]
    leaf_b_metrics = result["models"]["leaf_b"]["metrics"]
    assert leaf_a_metrics["train_accuracy"] == 1.0
    assert leaf_a_metrics["val_accuracy"] == 1.0
    assert leaf_a_metrics["test_accuracy"] == 1.0
    assert leaf_a_metrics["macro_f1"] == 1.0
    assert leaf_a_metrics["feature_dim"] == 2
    assert leaf_b_metrics["val_accuracy"] < 1.0

    assert result["ensemble_metrics"]["ensemble_macro_f1"] == 1.0
    assert result["weighted_ensemble_metrics"]["val_macro_f1"] >= 0.5
    assert result["final_selection"]["best_single_leaf"] == "leaf_a"
    assert result["final_selection"]["final_choice"] in {"best_single_leaf", "weighted_ensemble"}
    assert result["final_selection"]["selection_basis"] == "val_macro_f1"
    assert result["models"]["leaf_a"]["metrics"]["selection_score"] == result["models"]["leaf_a"]["metrics"]["val_macro_f1"]
    assert result["selection_predictions"]["selection_basis"] == "val_macro_f1"
    assert result["selection_predictions"]["final_choice_strategy"] in {"best_single_leaf", "weighted_ensemble"}
    assert result["selection_predictions"]["window_ids_val"].size == 0
    assert "feature_dim" in result["metrics_markdown"]
    assert "macro_f1" in result["metrics_markdown"]


def test_shallow_ml_agent_balanced_pool_records_selected_algorithm(monkeypatch):
    class DummyEstimator:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.asarray([0 if row[0] < 0.5 else 1 for row in X])

        def predict_proba(self, X):
            pred = self.predict(X)
            out = np.zeros((len(pred), 2), dtype=float)
            for idx, label in enumerate(pred):
                out[idx, int(label)] = 1.0
            return out

    monkeypatch.setattr(shallow_module, "_build_estimator", lambda algorithm: DummyEstimator())

    datasets = {
        "leaf_pool": {
            "X_train": np.array([[0.0], [0.1], [0.9], [1.0]]),
            "X_val": np.array([[0.05], [0.95]]),
            "X_test": np.array([[0.03], [0.97]]),
            "y_train": np.array([0, 0, 1, 1]),
            "y_val": np.array([0, 1]),
            "y_test": np.array([0, 1]),
        }
    }

    result = shallow_ml_agent(datasets, algorithm="balanced_pool", cv_folds=0)

    assert result["candidate_algorithms"]
    assert result["models"]["leaf_pool"]["selected_algorithm"] in result["candidate_algorithms"]
    assert result["final_selection"]["selection_basis"] == "val_macro_f1"
