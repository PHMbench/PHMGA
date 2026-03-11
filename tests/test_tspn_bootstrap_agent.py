from __future__ import annotations

from pathlib import Path

import numpy as np

from src.agents.tspn_bootstrap_agent import tspn_bootstrap_agent
from src.model.explainable.config_schema import TSPNConfig
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def test_tspn_bootstrap_agent_generates_valid_config(tmp_path: Path):
    # Two-channel minimal DAG with one processed node to provide token semantics.
    L = 256
    train = {"id1": np.random.randn(1, L, 1).astype(np.float32), "id2": np.random.randn(1, L, 1).astype(np.float32)}
    test = {"id3": np.random.randn(1, L, 1).astype(np.float32)}

    ch1 = InputData(node_id="ch1", parents=[], shape=(1, L, 1), results={"train": train, "test": test}, meta={"channel": "ch1"})
    ch2 = InputData(node_id="ch2", parents=[], shape=(1, L, 1), results={"train": train, "test": test}, meta={"channel": "ch2"})

    # A processed node to indicate an FFT op exists at depth 1.
    n1 = ProcessedData(
        node_id="fft_01_ch1",
        parents=["ch1"],
        shape=(1, L, 1),
        source_signal_id="ch1",
        method="fft",
        results={"train": train, "test": test},
        meta={"tool": "fft"},
    )

    dag = DAGState(user_instruction="demo", channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2, "fft_01_ch1": n1}, leaves=["fft_01_ch1", "ch2"])
    state = PHMState(
        case_name="demo",
        user_instruction="demo",
        reference_signal=ch1,
        test_signal=ch2,
        dag_state=dag,
        labels_train={"id1": "0", "id2": "1"},
        labels_test={"id3": "0"},
        save_dir=str(tmp_path),
        train_backend="tspn",
    )

    out = tspn_bootstrap_agent(state)
    assert "model_config_path" in out
    assert Path(out["model_config_path"]).exists()
    cfg = TSPNConfig.model_validate(out["current_model_config"])
    assert cfg.model.in_dim == L
    assert cfg.model.in_channels == 2
    assert cfg.model.num_classes >= 2
