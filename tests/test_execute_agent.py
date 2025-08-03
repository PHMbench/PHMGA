import os
import numpy as np
from langchain_community.chat_models import FakeListChatModel

from phm_core import PHMState, DAGState, InputData
from src.agents.execute_agent import execute_agent


def test_execute_agent_runs(tmp_path):
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    sig = np.ones((1, 4, 1))
    ch1 = InputData(node_id="ch1", data={"signal": sig}, results={"ref": sig, "tst": sig}, parents=[], shape=sig.shape)
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        detailed_plan=[{"op_name": "mean", "params": {"parent": "ch1"}}],
    )

    out = execute_agent(state)
    assert out["executed_steps"] == 1
    assert "mea_01_ch1" in state.dag_state.nodes
