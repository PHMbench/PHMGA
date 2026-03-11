import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.states.phm_states import PHMState, DAGState, InputData, ProcessedData
from src.agents.dataset_preparer_agent import dataset_preparer_agent, _find_root_label_maps


def test_dataset_preparer_agent(tmp_path):
    sig = np.array([1, 2, 3])
    train_path = tmp_path / "train.npy"
    val_path = tmp_path / "val.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, sig)
    np.save(val_path, sig + 1)
    np.save(test_path, sig + 2)
    ch1 = InputData(
        node_id="ch1",
        data={"signal": sig},
        results={"train": sig, "val": sig + 1, "test": sig + 2},
        parents=[],
        shape=sig.shape,
        meta={"labels_train": {"s1": 0}, "labels_val": {"s1": 0}, "labels_test": {"s1": 0}},
    )
    proc = ProcessedData(
        node_id="fft_01_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        processed_data=sig,
        results={"train": sig, "val": sig + 1, "test": sig + 2},
        meta={"saved": {"train_path": str(train_path), "val_path": str(val_path), "test_path": str(test_path)}},
        shape=sig.shape,
    )
    dag = DAGState(user_instruction="diagnose", channels=["ch1"], nodes={"ch1": ch1, "fft_01_ch1": proc}, leaves=["fft_01_ch1"])
    state = PHMState(user_instruction="diagnose", reference_signal=ch1, test_signal=ch1, dag_state=dag)
    out = dataset_preparer_agent(state, config={"stage": "processed", "flatten": True})
    assert "fft_01_ch1" in out["datasets"]
    ds = out["datasets"]["fft_01_ch1"]
    assert ds["X_train"].shape[0] == sig.size
    assert ds["X_val"].shape[0] == sig.size
    assert f"ds_fft_01_ch1" in state.dag_state.nodes


def test_find_root_label_maps_cycle_guard():
    p1 = ProcessedData(
        node_id="p1",
        parents=["p2"],
        source_signal_id="p2",
        method="fft",
        processed_data=np.array([1.0]),
        results={},
        meta={},
        shape=(1,),
    )
    p2 = ProcessedData(
        node_id="p2",
        parents=["p1"],
        source_signal_id="p1",
        method="fft",
        processed_data=np.array([1.0]),
        results={},
        meta={},
        shape=(1,),
    )
    errors = []
    labels_train, labels_val, labels_test = _find_root_label_maps(
        "p1",
        {"p1": p1, "p2": p2},
        max_hops=8,
        error_sink=errors,
    )
    assert labels_train == {}
    assert labels_val == {}
    assert labels_test == {}
    assert any("cycle" in msg for msg in errors)
