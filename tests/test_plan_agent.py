import os
import sys
from pathlib import Path
import numpy as np
from langchain_community.chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import PHMState, DAGState, InputData
from src.agents.plan_agent import plan_agent


def _make_state() -> PHMState:
    sig = np.ones((1, 4, 1))
    ch1 = InputData(node_id="ch1", data={"signal": sig}, parents=[], shape=sig.shape)
    dag = DAGState(user_instruction="demo", channels=["ch1"])
    return PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
    )


def test_plan_agent_generates_plan_from_dict_payload():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['{"plan": [{"op_name": "mean", "params": {"parent": "ch1"}}]}']
    )
    import src.tools  # ensure OP_REGISTRY populated

    state = _make_state()
    out = plan_agent(state)
    assert out["detailed_plan"], "plan should not be empty"
    assert out["detailed_plan"][0]["op_name"] == "mean"


def test_plan_agent_generates_plan_from_list_payload():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['[{"op_name": "mean", "params": {"parent": "ch1"}}]']
    )
    import src.tools  # ensure OP_REGISTRY populated

    state = _make_state()
    out = plan_agent(state)
    assert out["detailed_plan"], "plan should not be empty"
    assert out["detailed_plan"][0]["op_name"] == "mean"


def test_plan_agent_generates_plan_from_fenced_json():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['```json\n{"plan":[{"op_name":"mean","params":{"parent":"ch1"}}]}\n```']
    )
    import src.tools  # ensure OP_REGISTRY populated

    state = _make_state()
    out = plan_agent(state)
    assert out["detailed_plan"], "plan should not be empty"
    assert out["detailed_plan"][0]["op_name"] == "mean"


def test_plan_agent_bad_json_returns_empty_plan_and_error_log():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(responses=["NOT_JSON"])
    import src.tools  # ensure OP_REGISTRY populated

    state = _make_state()
    out = plan_agent(state)
    assert out["detailed_plan"] == []
    assert state.error_logs, "expected planner error log for invalid JSON payload"
