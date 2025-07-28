import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.configuration import Config
from src.graph import build_graph
from src.state import SignalData, PHMState


def create_state(use_patch: bool) -> PHMState:
    ref = SignalData(data=np.ones((2, 128, 3)).tolist(), sampling_rate=1)
    test = SignalData(data=np.ones((2, 128, 3)).tolist(), sampling_rate=1)
    return {
        "user_instruction": "test",
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


def test_end2end_patch():
    config = Config(use_patch=True, patch_size=64, signal_processing_methods=["id"], feature_methods=["mean"], similarity_method="l2", decision_model="simple", max_loops=1)
    state = create_state(True)
    graph = build_graph(config)
    result = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    proc = result["processed_signals"][0].processed_data
    arr = np.array(proc)
    assert arr.shape == (2, 2, 64, 3)
    assert result["final_report"]
