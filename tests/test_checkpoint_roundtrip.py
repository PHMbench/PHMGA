import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import Config
from src.graph import build_graph
from src.state import PHMState, SignalData
from src.database import SQLiteCheckpointer


def test_checkpoint_roundtrip(tmp_path):
    db_path = tmp_path / "db.sqlite"
    config = Config(
        signal_processing_methods=["identity"],
        feature_methods=["mean"],
        similarity_method="euclidean",
        decision_model="threshold",
    )
    graph = build_graph(config, db_path=str(db_path))
    ref = np.zeros((1, 10, 1))
    test = np.zeros((1, 10, 1))
    state: PHMState = {
        "user_instruction": "run",
        "reference_signal": SignalData(data=ref.tolist(), sampling_rate=100),
        "test_signal": SignalData(data=test.tolist(), sampling_rate=100),
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
    result = graph.invoke(state, config={"configurable": {"thread_id": "0"}})
    assert os.path.exists(db_path)
    snapshot = graph.get_state({"configurable": {"thread_id": "0"}})
    assert snapshot is not None
