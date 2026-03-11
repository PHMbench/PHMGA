from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import DAGState, InputData, PHMState
from src.agents.deep_model_train_agent import deep_model_train_agent


@pytest.mark.skipif(
    os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() not in {"1", "true", "yes", "y"},
    reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch training tests.",
)
def test_grad_clip_called_during_training(monkeypatch: pytest.MonkeyPatch, tmp_path):
    torch = pytest.importorskip("torch")

    model_cfg = {
        "model": {
            "name": "tspn",
            "device": "cpu",
            "num_classes": 2,
            "in_dim": 32,
            "in_channels": 1,
            "out_channels": 2,
            "scale": 2,
            "layers": [{"gate_temperature": 1.0, "ops": [{"token": "I"}, {"token": "WF"}]}],
            "features": ["Mean", "Std"],
            "disabled_ops": {},
        },
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "debug": True,
            "debug_epochs": 1,
            "debug_max_samples": 8,
            "val_ratio": 0.25,
            "grad_clip_norm": 0.5,
        },
        "explain": {"topk_ops": 2, "save_wavefilters": False},
    }
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_cfg), encoding="utf-8")

    rng = np.random.default_rng(0)
    ref_samples = {
        f"s{i}": rng.normal(size=(1, 32, 1)).astype(np.float32)
        for i in range(8)
    }
    ch1 = InputData(
        node_id="ch1",
        parents=[],
        shape=(1, 32, 1),
        results={"train": ref_samples, "test": {}},
        data={},
    )
    state = PHMState(
        case_name="clip_smoke",
        user_instruction="train",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=DAGState(user_instruction="train", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"]),
        labels_train={f"s{i}": ("A" if i % 2 == 0 else "B") for i in range(8)},
        train_backend="tspn",
        model_config_path=str(model_path),
        save_dir=str(tmp_path / "save"),
    )

    clip_calls = {"count": 0}
    original_clip = torch.nn.utils.clip_grad_norm_

    def _clip_wrapper(*args, **kwargs):
        clip_calls["count"] += 1
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _clip_wrapper)
    out = deep_model_train_agent(state)

    assert clip_calls["count"] > 0
    assert "ml_results" in out and "tspn" in out["ml_results"]
