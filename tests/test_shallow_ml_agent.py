import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.agents.shallow_ml_agent import shallow_ml_agent


def test_shallow_ml_agent_basic():
    rng = np.random.default_rng(0)
    datasets = {
        "fft_01": {
            "X_train": rng.random((20, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_test": rng.integers(0, 2, 10),
        },
        "psd_02": {
            "X_train": rng.random((20, 5)),
            "X_test": rng.random((10, 5)),
            "y_train": rng.integers(0, 2, 20),
            "y_test": rng.integers(0, 2, 10),
        },
    }

    out = shallow_ml_agent(datasets)

    assert len(out["models"]) == len(datasets)
    assert "accuracy" in out["ensemble_metrics"]
