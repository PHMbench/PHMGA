from __future__ import annotations

import numpy as np

from phm_core import DAGState, InputData, PHMState
from src.agents.execute_agent import execute_agent


def _base_state(plan):
    sig = np.random.randn(1, 256, 1).astype(np.float32)
    ch1 = InputData(
        node_id="ch1",
        data={"signal": sig},
        results={"ref": {"s1": sig}, "tst": {"s1": sig}},
        parents=[],
        shape=sig.shape,
        meta={},
    )
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    return PHMState(
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        detailed_plan=plan,
    )


def test_execute_unknown_op_is_tagged_as_unsupported():
    state = _base_state([{"op_name": "unknown_magic", "params": {"parent": "ch1"}}])
    out = execute_agent(state)
    assert out["executed_steps"] == 0
    assert any("[unsupported_op]" in line for line in out["dag_state"].error_log)


def test_execute_missing_dependency_is_categorized(monkeypatch):
    def _raise_import_error(*args, **kwargs):
        raise ImportError("Librosa is not installed.")

    monkeypatch.setattr("src.agents.execute_agent._execute_single_variable_op", _raise_import_error)
    state = _base_state([{"op_name": "mean", "params": {"parent": "ch1"}}])
    out = execute_agent(state)
    assert out["executed_steps"] == 0
    assert any("[missing_dep]" in line for line in out["dag_state"].error_log)


def test_execute_contract_violation_stops_current_round():
    state = _base_state(
        [
            {"op_name": "psd", "params": {"parent": "ch1"}},
            {"op_name": "mean", "params": {"parent": "ch1"}},
        ]
    )
    state.data_cfg = {
        "operator_contract": "rm101_closed_v1",
        "enforce_tspn_closed_world": True,
    }
    out = execute_agent(state)
    assert out["executed_steps"] == 0
    assert any("[contract_violation]" in line for line in out["dag_state"].error_log)
    assert "mea_02_ch1" not in out["dag_state"].nodes
