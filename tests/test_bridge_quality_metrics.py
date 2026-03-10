from __future__ import annotations

from src.model.explainable.bridge import DAG2ConfigAdapter
from src.states.phm_states import DAGState, InputData, ProcessedData


def test_bridge_quality_metrics_are_emitted():
    L = 128
    ch1 = InputData(node_id="ch1", parents=[], shape=(1, L, 1), results={}, meta={"fs": 12800.0})
    n1 = ProcessedData(
        node_id="fft_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        results={},
        meta={"params": {}},
        shape=(1, L, 1),
    )
    n2 = ProcessedData(
        node_id="sub_1",
        parents=["fft_1"],
        source_signal_id="ch1",
        method="subtract",
        results={},
        meta={"params": {}},
        shape=(1, L, 1),
    )
    dag = DAGState(
        user_instruction="bridge-quality-test",
        channels=["ch1"],
        nodes={"ch1": ch1, "fft_1": n1, "sub_1": n2},
        leaves=["sub_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=12800.0,
        max_layers=3,
        parallel_ops_per_layer=2,
        unsupported_policy="drop",
        min_effective_ops_ratio=0.4,
        enforce_tspn_closed_world=False,
    )
    bridge = adapter.adapt(dag)
    bridge_quality = (
        bridge.model_config.get("meta", {})
        .get("bridge", {})
        .get("bridge_quality", {})
    )
    assert isinstance(bridge_quality, dict)
    assert float(bridge_quality.get("effective_ops_ratio", -1.0)) >= 0.0
    assert int(bridge_quality.get("unsupported_nodes_count", 0)) >= 1
    assert int(bridge_quality.get("dropped_nodes_count", 0)) >= 1
    assert "bridge_quality" in bridge.init_metadata
    compile_quality = (bridge.init_metadata or {}).get("compile_quality") or {}
    assert compile_quality.get("operator_contract") == "rm101_closed_v1"
    assert bool(compile_quality.get("closed_world_pass")) is False
