import os
import numpy as np
from langchain_community.chat_models import FakeListChatModel

from phm_core import PHMState, DAGState, InputData
from src.agents.execute_agent import execute_agent


def test_execute_agent_fs_fallback_uses_sample_rate():
    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    sig1 = np.random.randn(1, 512, 1).astype(np.float32)
    sig2 = np.random.randn(1, 512, 1).astype(np.float32)

    ch1 = InputData(
        node_id="ch1",
        data={"signal": sig1},
        results={"ref": {"s1": sig1}, "tst": {"s1": sig1}},
        parents=[],
        shape=sig1.shape,
        meta={},
    )
    ch2 = InputData(
        node_id="ch2",
        data={"signal": sig2},
        results={"ref": {"s1": sig2}, "tst": {"s1": sig2}},
        parents=[],
        shape=sig2.shape,
        meta={},
    )
    dag = DAGState(
        user_instruction="demo",
        channels=["ch1", "ch2"],
        nodes={"ch1": ch1, "ch2": ch2},
        leaves=["ch1", "ch2"],
    )
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        detailed_plan=[{"op_name": "coherence", "params": {"parent": "ch1,ch2"}}],
        data_cfg={"sample_rate": 12000, "enforce_tspn_closed_world": False},
    )

    out = execute_agent(state)
    assert out["executed_steps"] == 1
    node = out["dag_state"].nodes["coh_01_ch1_ch2"]
    assert float(node.meta["params"]["fs"]) == 12000.0


def test_execute_agent_filter_defaults_when_missing_required_params():
    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    sig = np.random.randn(1, 512, 1).astype(np.float32)
    ch1 = InputData(
        node_id="ch1",
        data={"signal": sig},
        results={"ref": {"s1": sig}, "tst": {"s1": sig}},
        parents=[],
        shape=sig.shape,
        meta={},
    )
    dag = DAGState(
        user_instruction="demo",
        channels=["ch1"],
        nodes={"ch1": ch1},
        leaves=["ch1"],
    )
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        detailed_plan=[{"op_name": "filter", "params": {"parent": "ch1"}}],
        data_cfg={"sample_rate": 12800},
    )

    out = execute_agent(state)
    assert out["executed_steps"] == 1
    node = out["dag_state"].nodes["fil_01_ch1"]
    params = node.meta["params"]
    assert params["filter_type"] == "band"
    assert params["order"] == 4
    assert isinstance(params["cutoff"], (list, tuple))
    assert len(params["cutoff"]) == 2
    assert float(params["fs"]) == 12800.0


def test_execute_agent_skips_unknown_op_and_continues():
    os.environ["FAKE_LLM"] = "true"
    from src import model

    model._FAKE_LLM = FakeListChatModel(responses=["0"])

    sig = np.random.randn(1, 256, 1).astype(np.float32)
    ch1 = InputData(
        node_id="ch1",
        data={"signal": sig},
        results={"ref": {"s1": sig}, "tst": {"s1": sig}},
        parents=[],
        shape=sig.shape,
        meta={},
    )
    dag = DAGState(
        user_instruction="demo",
        channels=["ch1"],
        nodes={"ch1": ch1},
        leaves=["ch1"],
    )
    state = PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        detailed_plan=[
            {"op_name": "unknown_magic_op", "params": {"parent": "ch1"}},
            {"op_name": "mean", "params": {"parent": "ch1"}},
        ],
    )

    out = execute_agent(state)
    assert out["executed_steps"] == 1
    assert "mea_02_ch1" in out["dag_state"].nodes
    assert any("Skip step index=1" in item for item in out["dag_state"].error_log)
