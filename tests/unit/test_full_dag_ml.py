from __future__ import annotations

import numpy as np

from src.data.protocol import SignalRecord
from src.evaluation.full_dag_ml import evaluate_full_dag_leaves
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def _build_state() -> PHMState:
    ch1 = InputData(node_id="ch1", parents=[], shape=(1, 8, 1), results={}, meta={"channel": "ch1"})
    ch2 = InputData(node_id="ch2", parents=[], shape=(1, 8, 1), results={}, meta={"channel": "ch2"})
    mean_1 = ProcessedData(
        node_id="mean_01_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="mean",
        results=None,
        meta={"tool": "mean", "params": {"axis": -2}, "parent": "ch1"},
        shape=(1, 1),
    )
    mean_2 = ProcessedData(
        node_id="mean_02_ch2",
        parents=["ch2"],
        source_signal_id="ch2",
        method="mean",
        results=None,
        meta={"tool": "mean", "params": {"axis": -2}, "parent": "ch2"},
        shape=(1, 1),
    )
    concat = ProcessedData(
        node_id="concat_03_mean_01_ch1_mean_02_ch2",
        parents=["mean_01_ch1", "mean_02_ch2"],
        source_signal_id="mean_01_ch1,mean_02_ch2",
        method="concatenate",
        results=None,
        meta={
            "tool": "concatenate",
            "params": {"axis": -1},
            "parent": "mean_01_ch1,mean_02_ch2",
        },
        shape=(1, 2),
    )
    std_leaf = ProcessedData(
        node_id="std_04_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="std",
        results=None,
        meta={"tool": "std", "params": {"axis": -2}, "parent": "ch1"},
        shape=(1, 1),
    )
    dag = DAGState(
        user_instruction="rm101 evaluation",
        channels=["ch1", "ch2"],
        nodes={
            "ch1": ch1,
            "ch2": ch2,
            "mean_01_ch1": mean_1,
            "mean_02_ch2": mean_2,
            "concat_03_mean_01_ch1_mean_02_ch2": concat,
            "std_04_ch1": std_leaf,
        },
        leaves=["concat_03_mean_01_ch1_mean_02_ch2", "std_04_ch1"],
    )
    return PHMState(
        case_name="rm101_eval",
        user_instruction="rm101 evaluation",
        reference_signal=ch1,
        test_signal=ch2,
        dag_state=dag,
    )


def _signal_pair(label: int, offset: float) -> np.ndarray:
    base = 0.0 if label == 0 else 10.0
    channel_1 = np.array([base + offset, base + 0.1 + offset] * 4)
    channel_2 = np.array([base / 2.0 + offset, base / 2.0 + 0.2 + offset] * 4)
    return np.vstack([channel_1, channel_2])


def _build_split_records() -> dict[str, list[SignalRecord]]:
    outputs: dict[str, list[SignalRecord]] = {"train": [], "val": [], "test": []}
    offsets = {"train": 0.0, "val": 0.05, "test": 0.1}
    for split_name, offset in offsets.items():
        for label in (0, 1):
            for index in range(4):
                outputs[split_name].append(
                    SignalRecord(
                        source_sample_id=f"{split_name}_{label}_{index}",
                        window_id=f"{split_name}_{label}_{index}",
                        window_index=index,
                        split=split_name,  # type: ignore[arg-type]
                        label=label,
                        window=_signal_pair(label, offset + index * 0.01),
                    )
                )
    return outputs


def test_evaluate_full_dag_leaves_returns_leaf_metrics_and_selection():
    state = _build_state()
    split_records = _build_split_records()

    result = evaluate_full_dag_leaves(state, split_records=split_records, algorithm="RandomForest")

    dag_summary = result["dag_summary"]
    leaf_metrics = {row["leaf_id"]: row for row in result["leaf_metrics"]}
    final_selection = result["final_selection"]

    assert dag_summary["node_count"] == 6
    assert dag_summary["edge_count"] == 5
    assert set(dag_summary["unique_ops"]) == {"concatenate", "mean", "std"}

    assert set(leaf_metrics) == {"concat_03_mean_01_ch1_mean_02_ch2", "std_04_ch1"}
    assert leaf_metrics["concat_03_mean_01_ch1_mean_02_ch2"]["feature_dim"] == 2
    assert leaf_metrics["concat_03_mean_01_ch1_mean_02_ch2"]["val_macro_f1"] >= 0.99
    assert leaf_metrics["std_04_ch1"]["feature_dim"] == 1

    assert final_selection["best_single_leaf"]["leaf_id"] == "concat_03_mean_01_ch1_mean_02_ch2"
    assert final_selection["final_choice"]["strategy"] in {"best_single_leaf", "weighted_ensemble"}
    assert "concat_03_mean_01_ch1_mean_02_ch2" in result["metrics_markdown"]
