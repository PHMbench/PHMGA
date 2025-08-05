import os
import sys
import numpy as np
from langchain_community.chat_models import FakeListChatModel

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.states.phm_states import PHMState, DAGState, InputData
from src.agents.plan_agent import plan_agent
from src.agents.execute_agent import execute_agent


def test_plan_agent_injects_fs_for_spectral_operator():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['{"plan": [{"parent": "ch1", "op_name": "spectral_centroid", "params": {}}]}']
    )
    import src.tools  # ensure OP_REGISTRY populated

    sig = np.ones((1, 4, 1))
    ch1 = InputData(node_id="ch1", data={"signal": sig}, parents=[], shape=sig.shape)
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        fs=1000,
    )
    out = plan_agent(state)
    assert out["detailed_plan"][0]["params"]["fs"] == 1000


def test_execute_agent_adds_fs_if_missing(tmp_path):
    os.environ["FAKE_LLM"] = "true"
    os.environ["PHM_SAVE_DIR"] = str(tmp_path)
    from src import model
    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    spec = np.ones((1, 3, 1))
    ch1 = InputData(
        node_id="ch1",
        data={},
        results={"ref": spec, "tst": spec},
        parents=[],
        shape=spec.shape,
    )
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        fs=2000,
        detailed_plan=[{"op_name": "spectral_centroid", "params": {}, "parent": "ch1"}],
    )

    out = execute_agent(state)
    node_id = "spectral_centroid_01_ch1"
    assert out["dag_state"].nodes[node_id].meta["params"]["fs"] == 2000
