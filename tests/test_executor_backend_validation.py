import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.states.phm_states import DAGState, InputData, PHMState
import src.phm_outer_graph as outer_graph


def test_executor_invalid_train_backend_fails_fast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(outer_graph, "inquirer_agent", lambda state, metrics: {})
    monkeypatch.setattr(outer_graph, "dataset_preparer_agent", lambda state: {})
    monkeypatch.setattr(outer_graph, "dag_init_agent", lambda state: {})
    monkeypatch.setattr(outer_graph, "tspn_bootstrap_agent", lambda state: {})
    monkeypatch.setattr(outer_graph, "report_agent_node", lambda state: {})

    sig = np.zeros((1, 16, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", parents=[], data={"signal": sig}, shape=sig.shape)
    state = PHMState(
        user_instruction="test invalid backend",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=DAGState(user_instruction="test", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"]),
        train_backend="invalid_backend",
    )

    graph = outer_graph.build_executor_graph()
    with pytest.raises(ValueError, match="Invalid train_backend"):
        list(graph.stream(state, config={"configurable": {"thread_id": "test-backend-validation"}}))
