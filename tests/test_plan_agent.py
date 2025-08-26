import os
import numpy as np
from langchain_community.chat_models import FakeListChatModel

from src.states.phm_states import PHMState, DAGState, InputData
from src.agents.plan_agent import plan_agent


def test_plan_agent_generates_plan():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['{"plan": [{"op_name": "mean", "params": {"parent": "ch1"}}]}']
    )
    import src.tools  # ensure OP_REGISTRY populated

    sig = np.ones((1, 4, 1))
    ch1 = InputData(node_id="ch1", data={"signal": sig}, parents=[], shape=sig.shape)
    dag = DAGState(user_instruction="demo", channels=["ch1"])
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
    )
    out = plan_agent(state)
    assert out["detailed_plan"], "plan should not be empty"
    assert out["detailed_plan"][0]["op_name"] == "mean"
