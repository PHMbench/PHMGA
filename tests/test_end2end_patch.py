import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import Config
from src.graph import build_graph
from src.state import PHMState, SignalData


def test_end2end_patch(tmp_path):
    ref = np.random.randn(2, 100, 3)
    test = np.random.randn(2, 100, 3)
    ref_path = tmp_path / "ref.npy"
    test_path = tmp_path / "test.npy"
    np.save(ref_path, ref)
    np.save(test_path, test)

    config = Config(
        use_patch=True,
        patch_size=50,
        signal_processing_methods=["identity"],
        feature_methods=["mean"],
        similarity_method="euclidean",
        decision_model="threshold",
    )
    graph = build_graph(config, db_path=str(tmp_path / "db.sqlite"))

    state: PHMState = {
        "user_instruction": "analyze",
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
    processed = np.array(result["processed_signals"][0].processed_data)
    assert processed.shape == (2, 2, 50, 3)
    feats = result["extracted_features"][0].features
    assert len(feats) == 2
