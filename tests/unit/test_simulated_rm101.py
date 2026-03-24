from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from src.evaluation import compute_cross_dag_late_fusion, summarize_dag_state
from src.simulated_rm101 import (
    SIMULATED_MODEL_TAGS,
    apply_simulated_template,
    get_simulated_run_info,
    validate_complexity_ladder,
)
from src.states.phm_states import DAGState, InputData, PHMState


def _build_base_state() -> PHMState:
    nodes = {}
    leaves = []
    labels = {"ref_a": 0, "ref_b": 1, "test_a": 0, "test_b": 1}
    time = np.linspace(0.0, 1.0, 4096, endpoint=False)
    for index in range(1, 9):
        channel = f"ch{index}"
        signal_a = np.sin(2 * np.pi * index * time).reshape(1, -1, 1)
        signal_b = np.cos(2 * np.pi * index * time).reshape(1, -1, 1)
        node = InputData(
            node_id=channel,
            parents=[],
            shape=signal_a.shape,
            data={},
            results={
                "ref": {"ref_a": signal_a, "ref_b": signal_b},
                "tst": {"test_a": signal_a, "test_b": signal_b},
            },
            metadata={},
            meta={"channel": channel, "labels": labels, "fs": 3125.0},
        )
        nodes[channel] = node
        leaves.append(channel)
    dag = DAGState(
        user_instruction="simulate RM101 builders",
        channels=[f"ch{index}" for index in range(1, 9)],
        nodes=nodes,
        leaves=leaves,
    )
    return PHMState(
        case_name="exp2_rm101_paper",
        user_instruction="simulate RM101 builders",
        reference_signal=nodes["ch1"],
        test_signal=nodes["ch1"],
        dag_state=dag,
        fs=3125.0,
    )


def test_simulated_templates_form_complexity_ladder():
    summaries = {}
    states = {}
    for model_tag in SIMULATED_MODEL_TAGS:
        state = apply_simulated_template(_build_base_state(), model_tag)
        states[model_tag] = state
        summaries[model_tag] = summarize_dag_state(state)

    ladder = validate_complexity_ladder(states)

    assert ladder["google/gemini-2.0-flash-001"]["depth"] == 3
    assert ladder["google/gemini-2.5-flash"]["depth"] == 4
    assert ladder["google/gemini-2.5-pro"]["depth"] == 5
    assert ladder["google/gemini-2.0-flash-001"]["node_count"] == 20
    assert ladder["google/gemini-2.5-flash"]["node_count"] == 28
    assert ladder["google/gemini-2.5-pro"]["node_count"] == 52
    assert ladder["google/gemini-2.0-flash-001"]["node_count"] < ladder["google/gemini-2.5-flash"]["node_count"]
    assert ladder["google/gemini-2.5-flash"]["node_count"] < ladder["google/gemini-2.5-pro"]["node_count"]
    assert "fft" in summaries["google/gemini-2.0-flash-001"]["unique_ops"]
    assert "psd" in summaries["google/gemini-2.0-flash-001"]["unique_ops"]
    assert "order_track_resample" in summaries["google/gemini-2.5-flash"]["unique_ops"]
    assert "coherence" in summaries["google/gemini-2.5-flash"]["unique_ops"]
    assert "torque_normalize" in summaries["google/gemini-2.5-flash"]["unique_ops"]
    assert "patch" in summaries["google/gemini-2.5-pro"]["unique_ops"]
    assert "stft" in summaries["google/gemini-2.5-pro"]["unique_ops"]
    assert "tsa_cycle_average" in summaries["google/gemini-2.5-pro"]["unique_ops"]


def test_compute_cross_dag_late_fusion_consumes_selection_predictions(tmp_path: Path):
    window_ids_val = np.asarray(["v1", "v2", "v3"], dtype=object)
    window_ids_test = np.asarray(["t1", "t2", "t3"], dtype=object)
    y_val = np.asarray([0, 1, 0], dtype=int)
    y_test = np.asarray([0, 1, 1], dtype=int)
    run_dirs = []

    for idx, model_tag in enumerate(SIMULATED_MODEL_TAGS[:2], start=1):
        run_info = get_simulated_run_info(model_tag)
        run_dir = tmp_path / run_info.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "selection_basis": "val_macro_f1",
            "window_ids_val": window_ids_val,
            "window_ids_test": window_ids_test,
            "y_val": y_val,
            "y_test": y_test,
            "best_single_leaf": {
                "val_pred": np.asarray([0, 1, 0], dtype=int),
                "test_pred": np.asarray([0, 1, idx % 2], dtype=int),
                "val_proba": None,
                "test_proba": None,
                "val_macro_f1": 0.7 - (idx * 0.1),
            },
            "weighted_ensemble": {
                "val_pred": np.asarray([0, 1, 0], dtype=int),
                "test_pred": np.asarray([0, 1, 1], dtype=int),
                "val_proba": None,
                "test_proba": None,
                "val_macro_f1": 0.65 - (idx * 0.05),
            },
            "final_choice": {
                "strategy": "best_single_leaf",
                "val_pred": np.asarray([0, 1, 0], dtype=int),
                "test_pred": np.asarray([0, 1, idx % 2], dtype=int),
                "val_proba": None,
                "test_proba": None,
                "val_macro_f1": 0.7 - (idx * 0.1),
            },
            "final_choice_strategy": "best_single_leaf",
        }
        with (run_dir / "selection_predictions.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        manifest = {
            "provider": "simulated",
            "model": model_tag,
            "paper_label": run_info.paper_label,
            "run_type": "simulated_variant",
            "selection_predictions_path": str(run_dir / "selection_predictions.pkl"),
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        run_dirs.append(run_dir)

    fusion = compute_cross_dag_late_fusion(run_dirs)

    assert fusion["selection_basis"] == "val_macro_f1"
    assert fusion["fusion_method"] == "weighted_vote"
    assert len(fusion["weights"]) == 2
    assert fusion["n_test_windows"] == 3
    assert fusion["test_metrics"]["accuracy"] >= 2 / 3
