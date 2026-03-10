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


def test_fixed_ids_minimal_state_fails_with_actionable_error(tmp_path: Path):
    sig = np.zeros((1, 64, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", data={}, results={}, parents=[], shape=sig.shape, meta={})
    dag = DAGState(user_instruction="fixed_ids_minimal", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="fixed_ids_minimal",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        case_name="fixed_ids_minimal",
        save_dir=str(tmp_path / "save"),
        train_backend="tspn",
        labels_ref={"s0": "0", "s1": "1"},
        data_cfg={
            "source_mode": "fixed_ids",
            "backend": "fixed_ids",
            "state_save_mode": "minimal",
            "model_config_path": "config/model_tspn_basic.yaml",
        },
    )

    with pytest.raises(ValueError, match="state_save_mode=full"):
        deep_model_train_agent(state)


def test_vibench_minimal_state_can_enter_data_factory_path(monkeypatch, tmp_path: Path):
    sig = np.zeros((1, 64, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", data={}, results={}, parents=[], shape=sig.shape, meta={})
    dag = DAGState(user_instruction="vibench_minimal", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="vibench_minimal",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        case_name="vibench_minimal",
        save_dir=str(tmp_path / "save"),
        train_backend="tspn",
        data_cfg={
            "backend": "vibench",
            "source_mode": "vibench",
            "state_save_mode": "minimal",
        },
    )

    def _fake_vibench_train(_state, *, run_dir, data_cfg):
        return {
            "ml_results": {"tspn": {"path": "vibench", "state_save_mode": data_cfg.get("state_save_mode")}},
            "run_dir": str(run_dir),
        }

    monkeypatch.setattr("src.agents.deep_model_train_agent._train_with_vibench_factory", _fake_vibench_train)
    out = deep_model_train_agent(state)
    assert (out.get("ml_results") or {}).get("tspn", {}).get("path") == "vibench"
    assert (out.get("ml_results") or {}).get("tspn", {}).get("state_save_mode") == "minimal"


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_TORCH_TESTS=1 to enable torch training contract tests.")
def test_rm101_vibench_contract_for_dag_bridge(monkeypatch, tmp_path: Path):
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
    ch1 = InputData(node_id="ch1", data={"signal": sig}, results={"ref": {"s0": sig}}, parents=[], shape=sig.shape, meta={})
    dag = DAGState(user_instruction="contract", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    state = PHMState(
        user_instruction="contract",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        case_name="rm101_contract_test",
        save_dir=str(tmp_path / "save"),
        train_backend="tspn",
        data_cfg={
            "backend": "vibench",
            "dataset_name": "RM_101_THU_GEARBOX",
            "source_mode": "vibench",
            "use_dag_model_config": True,
            "enforce_tspn_closed_world": False,
            "use_class_weight": True,
            "epochs": 1,
            "debug": True,
            "debug_epochs": 1,
            "batch_size": 4,
            "device": "cpu",
            "model_config_path": "config/model_tspn_basic.yaml",
        },
    )

    out = deep_model_train_agent(state)
    run_dir = Path(out["run_dir"])

    resolve_info = json.loads((run_dir / "config_resolve.json").read_text(encoding="utf-8"))
    assert resolve_info.get("config_source_mode") == "dag_bridge"
    bridge_quality = resolve_info.get("bridge_quality", {})
    assert "effective_ops_ratio" in bridge_quality
    assert "compatibility_quality" in resolve_info
    assert "operator_contract" in resolve_info
    assert "closed_world_pass" in resolve_info
    assert "compile_quality" in resolve_info
    assert (run_dir / "compatibility_report.json").exists()
    assert (run_dir / "contract_violation_report.json").exists()
    assert (run_dir / "dag_compile_report.json").exists()

    manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert int(manifest.get("n_train_batches", 0)) > 0
    assert int(manifest.get("n_val_batches", 0)) > 0
    assert int(manifest.get("n_test_batches", 0)) > 0

    state.case_name = "rm101_contract_test_gate"
    state.data_cfg["enforce_tspn_closed_world"] = True
    out_gate = deep_model_train_agent(state)
    gate_run_dir = Path(out_gate["run_dir"])
    gate_ml = dict((out_gate.get("ml_results") or {}).get("tspn") or {})
    assert gate_ml.get("error")
    gate_resolve = json.loads((gate_run_dir / "config_resolve.json").read_text(encoding="utf-8"))
    assert gate_resolve.get("closed_world_pass") is False
    assert (gate_run_dir / "dag_compile_report.json").exists()
