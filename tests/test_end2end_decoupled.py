import os
import numpy as np

from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData
from src.schemas.insight_schema import AnalysisInsight


def _make_state():
    ref = np.ones((1, 32, 1))
    test = np.ones((1, 32, 1)) * 2
    ref_node = InputData(node_id="ref_n", data={"signal": ref}, parents=[], shape=ref.shape)
    test_node = InputData(node_id="test_n", data={"signal": test}, parents=[], shape=test.shape)
    dag_state = DAGState(
        user_instruction="diagnose",
        reference_root="ref_root",
        test_root="test_root",
        leaves=["ref_root", "test_root"],
    )
    return PHMState(
        user_instruction="diagnose",
        reference_signal=ref_node,
        test_signal=test_node,
        dag_state=dag_state,
    )


def test_end2end_decoupled(tmp_path):
    state = _make_state()
    graph = build_outer_graph()
    final_state = graph.invoke(state, {"configurable": {"thread_id": "dec"}})

    insights = final_state["insights"]
    assert insights and all(isinstance(i, AnalysisInsight) for i in insights)
    assert final_state["needs_revision"] is False
    assert "| Insight |" in final_state["final_report"]
    assert os.path.exists("final_dag.png")
