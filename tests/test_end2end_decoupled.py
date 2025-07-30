import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()


import numpy as np
from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData
from src.schemas.insight_schema import AnalysisInsight


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


def test_full_workflow():
    state = make_state()
    app = build_outer_graph()
    final_state = app.invoke(state, {"configurable": {"thread_id": "dec"}})

    assert isinstance(final_state["insights"], list)
    assert final_state["insights"] and isinstance(final_state["insights"][0], AnalysisInsight)
    assert final_state["needs_revision"] is False
    assert "| Insight | Severity | Nodes |" in final_state["final_report"]
    assert os.path.exists("final_dag.png")
if __name__ == "__main__":
    test_full_workflow()
    print("All tests passed!")
