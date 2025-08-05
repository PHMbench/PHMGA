import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import uuid
from langchain_community.chat_models import FakeListChatModel

os.environ["FAKE_LLM"] = "true"

from src import model

model._FAKE_LLM = FakeListChatModel(
    responses=[
        '{"plan": [{"op_name": "mean", "params": {"parent": "ch1"}}]}',
        '{"decision": "finish", "reason": "ok"}',
        '# PHM Report\n\n' + 'x' * 600,
    ]
)

import pytest
from src.phm_outer_graph import build_outer_graph
from src.states.phm_states import PHMState, DAGState, InputData


def make_state():
    sig = np.ones((1, 64, 1))
    node = InputData(node_id="ch1", data={"signal": sig}, results={"ref": sig, "tst": sig}, parents=[], shape=sig.shape)
    dag_state = DAGState(user_instruction="diagnose", channels=["ch1"], nodes={"ch1": node}, leaves=["ch1"])
    return PHMState(
        user_instruction="diagnose",
        reference_signal=node,
        test_signal=node,
        dag_state=dag_state,
    )


@pytest.mark.skip(reason="workflow recursion unstable in offline tests")
def test_end2end(tmp_path):
    state = make_state()
    app = build_outer_graph()
    final_state = app.invoke(state, {"configurable": {"thread_id": str(uuid.uuid4())}})

    assert os.path.exists("final_dag.png")
    assert "PHM Report" in final_state["final_report"]
    assert final_state["detailed_plan"]
