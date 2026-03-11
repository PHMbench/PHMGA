import uuid
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import DAGState, InputData, PHMState
import src.graph.builder_loop_graph as builder_graph_module
import src.phm_outer_graph as outer_graph


@pytest.mark.skipif(not getattr(outer_graph, "_LANGGRAPH_OK", False), reason="requires langgraph runtime")
def test_builder_graph_stops_at_max_builder_iterations(monkeypatch: pytest.MonkeyPatch):
    def _plan(_state):
        return {"detailed_plan": []}

    def _execute(_state):
        return {}

    def _reflect(state, stage="POST_EXECUTE"):
        _ = stage
        return {
            "needs_revision": True,
            "reflection_history": state.reflection_history + ["continue"],
            "iteration_count": int(state.iteration_count) + 1,
        }

    monkeypatch.setattr(builder_graph_module, "plan_agent", _plan)
    monkeypatch.setattr(builder_graph_module, "execute_agent", _execute)
    monkeypatch.setattr(builder_graph_module, "reflect_agent_node", _reflect)

    sig = np.zeros((1, 16, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", parents=[], data={"signal": sig}, shape=sig.shape)
    state = PHMState(
        user_instruction="test",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=DAGState(user_instruction="test", channels=["ch1"]),
        needs_revision=True,
        max_builder_iterations=3,
    )

    graph = outer_graph.build_builder_graph()
    events = list(graph.stream(state, config={"configurable": {"thread_id": str(uuid.uuid4())}}))
    reflect_events = [
        node_update for event in events for node, node_update in event.items() if node == "reflect"
    ]
    assert len(reflect_events) == 3
    assert all(update.get("iteration_count", 0) <= 3 for update in reflect_events)
