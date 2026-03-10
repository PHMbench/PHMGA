import os
import sys
import json
from pathlib import Path
import numpy as np
from langchain_community.chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import PHMState, DAGState, InputData
from src.agents.plan_agent import plan_agent, _build_tools_description


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


def test_plan_tools_description_excludes_missing_dependency_ops(monkeypatch):
    def _fake_dep(module_name: str) -> bool:
        if module_name in {"nolds", "antropy", "librosa", "skimage", "pywt"}:
            return False
        return True

    monkeypatch.setattr("src.agents.plan_agent._dependency_available", _fake_dep)
    desc = _build_tools_description()
    assert "op_name: approximate_entropy" not in desc
    assert "op_name: permutation_entropy" not in desc
    assert "op_name: power_to_db" not in desc
    assert "op_name: mel_spectrogram" not in desc
    assert "op_name: patch" not in desc
    assert "op_name: wavelet_transform" not in desc


def test_plan_agent_sanitizes_hallucinated_ops():
    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(
        responses=[
            json.dumps(
                {
                    "plan": [
                        {"op_name": "spectral_entropy", "params": {"parent": "ch1"}},
                        {"op_name": "unknown_magic_op", "params": {"parent": "ch1"}},
                        {"op_name": "mean", "params": {"parent": "ch1"}},
                    ]
                }
            )
        ]
    )
    import src.tools  # noqa: F401 - ensure OP_REGISTRY populated

    state = _make_state()
    out = plan_agent(state)
    op_names = [step["op_name"] for step in out["detailed_plan"]]
    assert "spectral_flatness" in op_names
    assert "mean" in op_names
    assert "unknown_magic_op" not in op_names
    assert any("Planner sanitize" in item for item in state.error_logs)


def test_plan_agent_rejects_out_of_contract_operator():
    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(
        responses=[json.dumps({"plan": [{"op_name": "psd", "params": {"parent": "ch1"}}]})]
    )
    import src.tools  # noqa: F401 - ensure OP_REGISTRY populated

    state = _make_state()
    state.data_cfg = {
        "operator_contract": "rm101_closed_v1",
        "enforce_tspn_closed_world": True,
    }
    out = plan_agent(state)
    assert out["detailed_plan"] == []
    assert any("contract_violation" in item for item in state.error_logs)
