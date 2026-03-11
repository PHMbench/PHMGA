from __future__ import annotations

from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.phm_outer_graph as outer
import src.graph.executor_tspn_graph as executor_graph_module
from src.states.phm_states import DAGState, InputData, PHMState


def _make_state() -> PHMState:
    signal = np.random.randn(1, 32, 1).astype(np.float32)
    ch1 = InputData(
        node_id="ch1",
        parents=[],
        shape=signal.shape,
        data={"signal": signal},
        results={"ref": {"id1": signal}, "tst": {"id2": signal}},
        meta={"channel": "ch1", "fs": 12000},
    )
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    return PHMState(
        case_name="route_test",
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        train_backend="tspn",
    )


def test_executor_graph_uses_tspn_fast_path(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        executor_graph_module,
        "inquirer_agent",
        lambda state, metrics: calls.append("inquire") or {"insights": []},
    )
    monkeypatch.setattr(
        executor_graph_module,
        "dataset_preparer_agent",
        lambda state, config=None: calls.append("prepare") or {"datasets": {}},
    )
    monkeypatch.setattr(
        executor_graph_module,
        "dag_init_agent",
        lambda state: calls.append("init_dag") or {"dag_state": state.dag_state},
    )
    monkeypatch.setattr(
        executor_graph_module,
        "tspn_bootstrap_agent",
        lambda state: calls.append("bootstrap") or {"model_config_path": "dummy.yaml", "current_model_config": {}},
    )
    monkeypatch.setattr(
        executor_graph_module,
        "deep_model_train_agent",
        lambda state: calls.append("train") or {"ml_results": {"tspn": {"metrics": {"val": {"val_acc": 1.0}}}}},
    )
    monkeypatch.setattr(
        executor_graph_module,
        "report_agent_node",
        lambda state: calls.append("report") or {"final_report": "ok"},
    )

    app = outer.build_executor_graph()
    state = _make_state()

    observed_nodes: list[str] = []
    for event in app.stream(state, config={"configurable": {"thread_id": "tspn-route"}}):
        observed_nodes.extend(event.keys())

    assert "inquire" not in observed_nodes
    assert "prepare" not in observed_nodes
    assert {"init_dag", "bootstrap", "train", "report"}.issubset(set(observed_nodes))
    assert "inquire" not in calls
    assert "prepare" not in calls
    assert "train" in calls
