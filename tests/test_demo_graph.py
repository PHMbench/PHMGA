import numpy as np
from src.phm_demo import build_demo_graph, generate_signals


def test_demo_graph_runs():
    graph = build_demo_graph()
    state = generate_signals()
    result = graph.invoke(state)
    assert isinstance(result["output"], str)
    assert result["result"]
