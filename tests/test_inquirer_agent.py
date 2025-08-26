import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.states.phm_states import PHMState, DAGState, InputData, ProcessedData
from src.agents.inquirer_agent import inquirer_agent

def make_state():
    rng = np.random.default_rng(0)
    r1 = rng.random(8)
    t1 = rng.random(8)
    r2 = rng.random(8)
    t2 = rng.random(8)
    ref_fft_ch1 = ProcessedData(node_id="fft_ref_ch1", parents=["ref_ch1"], source_signal_id="ref_ch1", method="fft", processed_data=r1, results={"ref": r1}, meta={"channel": "ch1", "method": "fft", "kind": "ref"}, shape=r1.shape)
    tst_fft_ch1 = ProcessedData(node_id="fft_tst_ch1", parents=["tst_ch1"], source_signal_id="tst_ch1", method="fft", processed_data=t1, results={"tst": t1}, meta={"channel": "ch1", "method": "fft", "kind": "tst"}, shape=t1.shape)
    ref_fft_ch2 = ProcessedData(node_id="fft_ref_ch2", parents=["ref_ch2"], source_signal_id="ref_ch2", method="fft", processed_data=r2, results={"ref": r2}, meta={"channel": "ch2", "method": "fft", "kind": "ref"}, shape=r2.shape)
    tst_fft_ch2 = ProcessedData(node_id="fft_tst_ch2", parents=["tst_ch2"], source_signal_id="tst_ch2", method="fft", processed_data=t2, results={"tst": t2}, meta={"channel": "ch2", "method": "fft", "kind": "tst"}, shape=t2.shape)
    nodes = {n.node_id: n for n in [ref_fft_ch1, tst_fft_ch1, ref_fft_ch2, tst_fft_ch2]}
    dag = DAGState(user_instruction="demo", channels=["ref_ch1","tst_ch1","ref_ch2","tst_ch2"], nodes=nodes, leaves=list(nodes.keys()))
    state = PHMState(user_instruction="demo", reference_signal=InputData(node_id="r", data={}, parents=[], shape=(0,)), test_signal=InputData(node_id="t", data={}, parents=[], shape=(0,)), dag_state=dag)
    return state

def test_inquirer_agent_generates_similarity_nodes():
    state = make_state()
    result = inquirer_agent(state, metrics=["cosine"])
    assert len(result["new_nodes"]) == 2
    for nid in result["new_nodes"]:
        node = state.dag_state.nodes[nid]
        val = node.results["sim"]
        assert -1.0 <= val <= 1.0
