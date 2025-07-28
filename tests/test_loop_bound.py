import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.configuration import Config
from src.graph import build_graph
from src.state import SignalData, PHMState


def test_loop_bound():
    ref = SignalData(data=np.zeros((1, 10, 1)).tolist(), sampling_rate=1)
    test = SignalData(data=np.zeros((1, 10, 1)).tolist(), sampling_rate=1)
    state: PHMState = {
        "user_instruction": "instr",
        "reference_signal": ref,
        "test_signal": test,
        "plan": {},
        "reflection_history": [],
        "is_sufficient": False,
        "iteration_count": 0,
        "processed_signals": [],
        "extracted_features": [],
        "analysis_results": [],
        "final_decision": "",
        "final_report": "",
    }
    config = Config(max_loops=1, signal_processing_methods=["id"], feature_methods=["mean"], similarity_method="l2", decision_model="simple")
    graph = build_graph(config)
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert result["iteration_count"] == 1
