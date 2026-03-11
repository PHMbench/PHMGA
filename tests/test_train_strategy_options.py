from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from phm_core import DAGState, InputData, PHMState
from src.agents.deep_model_train_agent import deep_model_train_agent


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_TORCH_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch strategy tests.")
def test_vibench_train_strategy_options_are_recorded(monkeypatch, tmp_path: Path):
    torch = pytest.importorskip("torch")

    class _TinyLoader:
        def __init__(self, batches):
            self._batches = list(batches)
            self.dataset = [0] * int(sum(int(b["x"].shape[0]) for b in self._batches))

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    class _FakeBuild:
        def __init__(self):
            def _batch(seed: int):
                gen = torch.Generator().manual_seed(seed)
                x = torch.randn(4, 128, 2, generator=gen)
                y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
                return {"x": x, "y": y, "file_id": [f"s{seed}_{i}" for i in range(4)]}

            self.train_loader = _TinyLoader([_batch(1), _batch(2)])
            self.val_loader = _TinyLoader([_batch(3)])
            self.test_loader = _TinyLoader([_batch(4)])
            self.label_to_index = {"0": 0, "1": 1}

    class _FakeFactory:
        def __init__(self, cfg):
            self.cfg = cfg

        def build(self):
            return _FakeBuild()

    monkeypatch.setattr("src.agents.deep_model_train_agent.PHMVibenchDataFactory", _FakeFactory, raising=False)
    monkeypatch.setattr("src.utils.data_factory_wrapper.PHMVibenchDataFactory", _FakeFactory, raising=False)

    sig = np.zeros((1, 128, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", data={"signal": sig}, results={"train": {"s0": sig}}, parents=[], shape=sig.shape, meta={})
    dag = DAGState(user_instruction="strategy", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="strategy",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        case_name="strategy_test",
        save_dir=str(tmp_path / "save"),
        train_backend="tspn",
        model_config_path="config/model_tspn_basic.yaml",
        data_cfg={
            "backend": "vibench",
            "dataset_name": "RM_101_THU_GEARBOX",
            "source_mode": "vibench",
            "use_dag_model_config": True,
            "enforce_tspn_closed_world": False,
            "epochs": 2,
            "patience": 2,
            "debug": True,
            "debug_epochs": 1,
            "batch_size": 4,
            "device": "cpu",
            "scheduler": "plateau",
            "label_smoothing": 0.1,
            "early_stop_metric": "val_acc",
            "use_weighted_sampler": False,
        },
    )

    out = deep_model_train_agent(state)
    run_dir = Path(out["run_dir"])
    resolve_info = json.loads((run_dir / "config_resolve.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert resolve_info.get("scheduler") == "plateau"
    assert float(resolve_info.get("label_smoothing")) == pytest.approx(0.1)
    assert resolve_info.get("early_stop_metric") == "val_acc"
    strategy = metrics.get("train_strategy", {})
    assert strategy.get("scheduler") == "plateau"
    assert float(strategy.get("label_smoothing", 0.0)) == pytest.approx(0.1)
    assert strategy.get("early_stop_metric") == "val_acc"
