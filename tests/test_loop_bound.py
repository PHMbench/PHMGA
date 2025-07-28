import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import Config
from src.graph import build_graph
from src.state import PHMState, SignalData


def test_loop_bound(tmp_path):
    ref = np.zeros((1, 10, 1))
    test = np.zeros((1, 10, 1))
    config = Config(
        max_loops=1,
        signal_processing_methods=["identity"],
        feature_methods=["mean"],
        similarity_method="euclidean",
        decision_model="threshold",
    )
    graph = build_graph(config, db_path=str(tmp_path / "db.sqlite"))
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
    assert result["iteration_count"] <= 1
