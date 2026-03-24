from __future__ import annotations

import numpy as np

from src.data.protocol import SignalRecord
from src.evaluation.replay import (
    build_input_split_results,
    build_split_labels,
    replay_state_on_split_results,
)
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


def _signal_pair(label: int) -> np.ndarray:
    if label == 0:
        return np.vstack(
            [
                np.array([0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1]),
                np.array([1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1]),
            ]
        )
    return np.vstack(
        [
            np.array([10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1]),
            np.array([5.0, 5.1, 5.0, 5.1, 5.0, 5.1, 5.0, 5.1]),
        ]
    )


def _build_split_records() -> dict[str, list[SignalRecord]]:
    outputs: dict[str, list[SignalRecord]] = {"train": [], "val": [], "test": []}
    for split_name in outputs:
        for label in (0, 1):
            for index in range(3):
                outputs[split_name].append(
                    SignalRecord(
                        source_sample_id=f"{split_name}_{label}_{index}",
                        window_id=f"{split_name}_{label}_{index}",
                        window_index=index,
                        split=split_name,  # type: ignore[arg-type]
                        label=label,
                        window=_signal_pair(label),
                    )
                )
    return outputs


def test_build_input_split_results_slices_per_channel():
    split_records = _build_split_records()

    inputs = build_input_split_results(split_records, channel_ids=["ch1", "ch2"])
    labels = build_split_labels(split_records)

    assert set(inputs) == {"ch1", "ch2"}
    assert set(inputs["ch1"]) == {"train", "val", "test"}
    assert inputs["ch1"]["train"]["train_0_0"].shape == (1, 8, 1)
    assert inputs["ch2"]["val"]["val_1_2"].shape == (1, 8, 1)
    assert labels["test"]["test_1_0"] == 1


def test_replay_state_on_split_results_populates_leaf_results():
    state = _build_state()
    split_records = _build_split_records()
    inputs = build_input_split_results(split_records, channel_ids=state.dag_state.channels)

    replayed = replay_state_on_split_results(state, inputs)
    concat = replayed.state.dag_state.nodes["concat_03_mean_01_ch1_mean_02_ch2"]
    std_leaf = replayed.state.dag_state.nodes["std_04_ch1"]

    assert replayed.split_keys == ("train", "val", "test")
    assert set(concat.results) == {"train", "val", "test"}
    assert set(std_leaf.results["train"]) == {f"train_{label}_{index}" for label in (0, 1) for index in range(3)}

    concat_sample = concat.results["test"]["test_1_0"]
    std_sample = std_leaf.results["val"]["val_0_0"]
    assert concat_sample.shape == (1, 2)
    assert std_sample.shape == (1, 1)
    assert float(concat_sample[0, 0]) > 9.0
