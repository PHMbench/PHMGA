from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.explainable.bridge import DAG2ConfigAdapter
from src.model.explainable.config_schema import TSPNConfig
from src.states.phm_states import DAGState, InputData, ProcessedData


def test_dag2tspn_extended_mapping_and_metadata():
    L = 256
    ch1 = InputData(node_id="ch1", parents=[], shape=(1, L, 1), results={}, meta={"fs": 1000.0})

    n_norm = ProcessedData(
        node_id="norm_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="normalize",
        results={},
        meta={"params": {"method": "z_score"}},
        shape=(1, L, 1),
    )
    n_dt = ProcessedData(
        node_id="dt_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="detrend",
        results={},
        meta={"params": {"type": "linear"}},
        shape=(1, L, 1),
    )
    n_stft = ProcessedData(
        node_id="stft_1",
        parents=["norm_1"],
        source_signal_id="ch1",
        method="stft",
        results={},
        meta={"params": {"nperseg": 128, "noverlap": 64}},
        shape=(1, 64, 8),
    )
    n_unsupported = ProcessedData(
        node_id="sub_1",
        parents=["norm_1", "dt_1"],
        source_signal_id="ch1",
        method="subtract",
        results={},
        meta={"params": {}},
        shape=(1, L, 1),
    )

    dag = DAGState(
        user_instruction="extended mapping test",
        channels=["ch1"],
        nodes={
            "ch1": ch1,
            "norm_1": n_norm,
            "dt_1": n_dt,
            "stft_1": n_stft,
            "sub_1": n_unsupported,
        },
        leaves=["stft_1", "sub_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=1000.0,
        max_layers=3,
        parallel_ops_per_layer=2,
        out_channels=2,
        scale=2,
        feature_tokens=["Mean", "Std", "RMS"],
        preserve_dag_topology=True,
        allow_duplicate_tokens=True,
        unsupported_policy="fallback_to_identity",
    )
    result = adapter.adapt(dag)
    cfg = TSPNConfig.model_validate(result.model_config)

    # Layer-1 should keep normalize/detrend semantics.
    layer1_tokens = [op.token for op in cfg.model.layers[0].ops]
    assert "NORM" in layer1_tokens or "DT" in layer1_tokens

    # Layer-2 should include STFT with translated params.
    layer2 = cfg.model.layers[1]
    stft_ops = [op for op in layer2.ops if op.token == "STFT"]
    assert stft_ops, "Expected STFT token in second layer."
    assert stft_ops[0].params.get("n_fft") == 128
    assert stft_ops[0].params.get("hop_length") == 64

    bridge_meta = cfg.meta.get("bridge", {})
    assert bridge_meta.get("mapping_version")
    unsupported = bridge_meta.get("unsupported_nodes", [])
    assert any(item.get("method") == "subtract" for item in unsupported)
