from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.runtime.feature_plan import FeatureBranch, FeaturePlan, build_feature_plan
from src.runtime.pipeline import _run_ml_backend, _run_torch_backend
from src.states.phm_states import DAGState, InputData, ProcessedData


def _input_node(node_id: str, value: np.ndarray) -> InputData:
    return InputData(
        node_id=node_id,
        results={"ref": {"preview_ref": value}, "tst": {"preview_tst": value}},
        parents=[],
        shape=value.shape,
        meta={"channel": node_id, "fs": 25600},
    )


def _processed_node(node_id: str, parent: str, method: str, value: np.ndarray) -> ProcessedData:
    return ProcessedData(
        node_id=node_id,
        parents=[parent],
        source_signal_id=parent,
        method=method,
        results={"ref": {"preview_ref": value}, "tst": {"preview_tst": value}},
        meta={"tool": method, "params": {}, "method": method},
        shape=value.shape,
    )


def test_feature_plan_rejects_raw_transform_leaves():
    preview = np.ones((1, 64, 1), dtype=float)
    fft_preview = np.ones((1, 33, 1), dtype=float)
    dag = DAGState(
        user_instruction="fixture",
        channels=["ch1"],
        nodes={
            "ch1": _input_node("ch1", preview),
            "fft_01_ch1": _processed_node("fft_01_ch1", "ch1", "fft", fft_preview),
        },
        leaves=["fft_01_ch1"],
    )

    with pytest.raises(ValueError):
        build_feature_plan(dag)


def test_feature_plan_selects_terminal_aggregate_feature_leaves():
    preview = np.ones((1, 64, 1), dtype=float)
    fft_preview = np.ones((1, 33, 1), dtype=float)
    mean_preview = np.ones((1, 1), dtype=float)
    dag = DAGState(
        user_instruction="fixture",
        channels=["ch1"],
        nodes={
            "ch1": _input_node("ch1", preview),
            "fft_01_ch1": _processed_node("fft_01_ch1", "ch1", "fft", fft_preview),
            "mean_02_fft_01_ch1": _processed_node("mean_02_fft_01_ch1", "fft_01_ch1", "mean", mean_preview),
        },
        leaves=["mean_02_fft_01_ch1"],
    )

    plan = build_feature_plan(dag)

    assert len(plan.branches) == 1
    assert plan.branches[0].node_id == "mean_02_fft_01_ch1"
    assert plan.branches[0].dimension == 1


def test_ml_backend_outputs_normalized_weights_and_beats_uniform_baseline():
    feature_plan = FeaturePlan(
        branches=[
            FeatureBranch(node_id="good", op_name="mean", parents=["fft"], dimension=1, preview_shape=[1, 1]),
            FeatureBranch(node_id="bad", op_name="mean", parents=["fft"], dimension=1, preview_shape=[1, 1]),
        ]
    )
    feature_views = {
        "train": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["a", "b", "c", "d"],
            "branches": {
                "good": np.asarray([[-2.0], [-1.0], [1.0], [2.0]]),
                "bad": np.asarray([[0.1], [0.2], [0.1], [0.2]]),
            },
        },
        "val": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["e", "f", "g", "h"],
            "branches": {
                "good": np.asarray([[-2.2], [-1.2], [1.1], [2.1]]),
                "bad": np.asarray([[0.3], [0.2], [0.3], [0.2]]),
            },
        },
        "test": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["i", "j", "k", "l"],
            "branches": {
                "good": np.asarray([[-2.1], [-1.1], [1.2], [2.2]]),
                "bad": np.asarray([[0.2], [0.1], [0.2], [0.1]]),
            },
        },
    }

    result = _run_ml_backend(feature_plan, feature_views)

    assert pytest.approx(sum(result["branch_weights"].values()), rel=1e-6) == 1.0
    assert result["branch_weights"]["good"] > result["branch_weights"]["bad"]

    uniform_proba = None
    for branch_id in ("good", "bad"):
        estimator = LogisticRegression(max_iter=200, random_state=0)
        estimator.fit(feature_views["train"]["branches"][branch_id], feature_views["train"]["labels"])
        probs = estimator.predict_proba(feature_views["test"]["branches"][branch_id])
        uniform_proba = probs / 2.0 if uniform_proba is None else uniform_proba + probs / 2.0
    uniform_pred = np.argmax(uniform_proba, axis=1)
    uniform_f1 = float(f1_score(feature_views["test"]["labels"], uniform_pred, average="macro", zero_division=0))

    assert result["metrics"]["test"]["macro_f1"] >= uniform_f1


def test_torch_backend_trains_cpu_fusion_head_and_emits_weights():
    feature_plan = FeaturePlan(
        branches=[
            FeatureBranch(node_id="good", op_name="mean", parents=["fft"], dimension=1, preview_shape=[1, 1]),
            FeatureBranch(node_id="bad", op_name="mean", parents=["fft"], dimension=1, preview_shape=[1, 1]),
        ]
    )
    feature_views = {
        "train": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["a", "b", "c", "d"],
            "branches": {
                "good": np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=float),
                "bad": np.asarray([[0.1], [0.2], [0.1], [0.2]], dtype=float),
            },
        },
        "val": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["e", "f", "g", "h"],
            "branches": {
                "good": np.asarray([[-2.2], [-1.2], [1.1], [2.1]], dtype=float),
                "bad": np.asarray([[0.3], [0.2], [0.3], [0.2]], dtype=float),
            },
        },
        "test": {
            "labels": np.asarray([0, 0, 1, 1]),
            "sample_ids": ["i", "j", "k", "l"],
            "branches": {
                "good": np.asarray([[-2.1], [-1.1], [1.2], [2.2]], dtype=float),
                "bad": np.asarray([[0.2], [0.1], [0.2], [0.1]], dtype=float),
            },
        },
    }

    result = _run_torch_backend(
        feature_plan,
        feature_views,
        {"mode": "learned_softmax", "epochs": 3, "learning_rate": 0.05},
    )

    assert pytest.approx(sum(result["branch_weights"].values()), rel=1e-6) == 1.0
    assert len(result["training_curve"]) == 3
    assert set(result["metrics"]) == {"train", "val", "test"}
