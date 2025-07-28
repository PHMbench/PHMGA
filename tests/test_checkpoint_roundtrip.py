import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from pathlib import Path
from src.configuration import Config
from src.graph import build_graph
from src.state import SignalData, PHMState


def test_checkpoint_roundtrip(tmp_path: Path):
    db = tmp_path / "chk.db"
    ref = SignalData(data=np.zeros((1, 4, 1)).tolist(), sampling_rate=1)
    test = SignalData(data=np.zeros((1, 4, 1)).tolist(), sampling_rate=1)
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
    graph = build_graph(config, db_path=str(db))
    result1 = graph.invoke(state, {"configurable": {"thread_id": "t"}})
    assert db.exists()

    graph2 = build_graph(config, db_path=str(db))
    result2 = graph2.invoke(state, {"configurable": {"thread_id": "t2"}})
    assert result2["final_report"]
