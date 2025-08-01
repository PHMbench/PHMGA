import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData


def make_state():
    ref = np.ones((1, 64, 1))
    test = np.ones((1, 64, 1)) * 2
    ref_node = InputData(node_id="ref_node", data={"signal": ref}, parents=[], shape=ref.shape)
    test_node = InputData(node_id="test_node", data={"signal": test}, parents=[], shape=test.shape)
    dag_state = DAGState(user_instruction="diagnose", reference_root="ref_root", test_root="test_root", leaves=["ref_root", "test_root"])
    return PHMState(
        user_instruction="diagnose",
        reference_signal=ref_node,
        test_signal=test_node,
        dag_state=dag_state,
    )


def test_end2end(tmp_path):
    state = make_state()
    app = build_outer_graph()
    final_state = app.invoke(state, {"configurable": {"thread_id": "test"}})

    assert os.path.exists("final_dag.png")
    assert "PHM Report" in final_state["final_report"]
    assert final_state["detailed_plan"]
