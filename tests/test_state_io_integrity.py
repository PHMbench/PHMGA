from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import DAGState, InputData, PHMState
from src.utils import load_state, save_state


def _make_state(*, source_mode: str = "fixed_ids") -> PHMState:
    sig = np.ones((1, 16, 1), dtype=np.float32)
    ch1 = InputData(
        node_id="ch1",
        parents=[],
        shape=sig.shape,
        data={"signal": sig.copy()},
        results={
            "train": {"s_train": sig.copy()},
            "test": {"s_test": (sig * 2.0).copy()},
        },
    )
    dag = DAGState(user_instruction="io", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    return PHMState(
        case_name="io_case",
        user_instruction="io",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        data_cfg={
            "source_mode": source_mode,
            "backend": "vibench" if source_mode == "vibench" else "fixed_ids",
        },
    )


def _read_meta(state_path: Path) -> dict:
    meta_path = Path(f"{state_path}.meta.json")
    assert meta_path.exists()
    return json.loads(meta_path.read_text(encoding="utf-8"))


def test_state_save_and_load_with_checksum(tmp_path):
    state = _make_state()
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))
    assert state_path.exists()
    assert (tmp_path / "state.pkl.sha256").exists()

    loaded = load_state(str(state_path))
    assert loaded is not None
    assert loaded.case_name == state.case_name


def test_state_checksum_tamper_detected(tmp_path):
    state = _make_state()
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))

    with open(state_path, "ab") as f:
        f.write(b"TAMPER")

    loaded = load_state(str(state_path))
    assert loaded is None


def test_state_missing_sidecar_rejected_by_default(tmp_path, monkeypatch):
    state = _make_state()
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))
    os.remove(f"{state_path}.sha256")

    monkeypatch.delenv("PHM_ALLOW_UNVERIFIED_STATE", raising=False)
    loaded = load_state(str(state_path))
    assert loaded is None


def test_state_missing_sidecar_allowed_with_env(tmp_path, monkeypatch):
    state = _make_state()
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))
    os.remove(f"{state_path}.sha256")

    monkeypatch.setenv("PHM_ALLOW_UNVERIFIED_STATE", "1")
    loaded = load_state(str(state_path))
    assert loaded is not None


def test_state_save_mode_auto_vibench_uses_minimal(tmp_path):
    state = _make_state(source_mode="vibench")
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))
    meta = _read_meta(state_path)
    assert meta.get("effective_mode") == "minimal"
    assert int(meta.get("numpy_bytes_before", 0)) > int(meta.get("numpy_bytes_after", -1))

    loaded = load_state(str(state_path))
    assert loaded is not None
    root = loaded.dag_state.nodes["ch1"]
    assert root.data == {}
    assert root.results == {}


def test_state_save_mode_auto_fixed_ids_uses_full(tmp_path):
    state = _make_state(source_mode="fixed_ids")
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path))
    meta = _read_meta(state_path)
    assert meta.get("effective_mode") == "full"

    loaded = load_state(str(state_path))
    assert loaded is not None
    root = loaded.dag_state.nodes["ch1"]
    assert isinstance((root.results or {}).get("train"), dict)
    assert (root.results or {}).get("train")


def test_state_save_mode_full_overrides_vibench(tmp_path):
    state = _make_state(source_mode="vibench")
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path), save_mode="full", source_mode="vibench")
    meta = _read_meta(state_path)
    assert meta.get("effective_mode") == "full"

    loaded = load_state(str(state_path))
    assert loaded is not None
    root = loaded.dag_state.nodes["ch1"]
    assert isinstance((root.results or {}).get("train"), dict)
    assert (root.results or {}).get("train")


def test_state_save_mode_minimal_allowed_for_fixed_ids(tmp_path):
    state = _make_state(source_mode="fixed_ids")
    state_path = tmp_path / "state.pkl"
    assert save_state(state, str(state_path), save_mode="minimal", source_mode="fixed_ids")
    meta = _read_meta(state_path)
    assert meta.get("effective_mode") == "minimal"

    loaded = load_state(str(state_path))
    assert loaded is not None
    root = loaded.dag_state.nodes["ch1"]
    assert root.results == {}
