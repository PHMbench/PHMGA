import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.states.phm_states import PHMState, DAGState, InputData, ProcessedData
from src.agents.dataset_preparer_agent import dataset_preparer_agent


def test_dataset_preparer_agent(tmp_path):
    sig = np.array([1, 2, 3])
    ref_path = tmp_path / "ref.npy"
    tst_path = tmp_path / "tst.npy"
    np.save(ref_path, sig)
    np.save(tst_path, sig + 1)
    ch1 = InputData(node_id="ch1", data={"signal": sig}, results={"ref": sig, "tst": sig}, parents=[], shape=sig.shape)
    proc = ProcessedData(
        node_id="fft_01_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        processed_data=sig,
        results={"ref": sig, "tst": sig + 1},
        meta={"saved": {"ref_path": str(ref_path), "tst_path": str(tst_path)}},
        shape=sig.shape,
    )
    dag = DAGState(user_instruction="diagnose", channels=["ch1"], nodes={"ch1": ch1, "fft_01_ch1": proc}, leaves=["fft_01_ch1"])
    state = PHMState(user_instruction="diagnose", reference_signal=ch1, test_signal=ch1, dag_state=dag)
    out = dataset_preparer_agent(state, config={"stage": "processed", "flatten": True})
    assert "fft_01_ch1" in out["datasets"]
    ds = out["datasets"]["fft_01_ch1"]
    assert ds["X_train"].shape[0] == sig.size
    assert f"ds_fft_01_ch1" in state.dag_state.nodes
