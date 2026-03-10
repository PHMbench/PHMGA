from __future__ import annotations

import pytest

from src.model.explainable.bridge import DAG2ConfigAdapter
from src.states.phm_states import DAGState, InputData, ProcessedData


def _base_input(length: int = 128) -> InputData:
    return InputData(node_id="ch1", parents=[], shape=(1, length, 1), results={}, meta={"fs": 12800.0})


def test_rm101_strict_routes_aggregate_to_features_and_keeps_effective_ops():
    L = 128
    ch1 = _base_input(L)
    n_fft = ProcessedData(
        node_id="fft_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        results={},
        meta={"params": {}},
        shape=(1, L // 2 + 1, 1),
    )
    n_mean = ProcessedData(
        node_id="mean_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="mean",
        results={},
        meta={"params": {}},
        shape=(1, 1),
    )
    dag = DAGState(
        user_instruction="rm101 strict bridge check",
        channels=["ch1"],
        nodes={"ch1": ch1, "fft_1": n_fft, "mean_1": n_mean},
        leaves=["fft_1", "mean_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=12800.0,
        max_layers=2,
        parallel_ops_per_layer=2,
        unsupported_policy="drop",
        compat_profile="rm101_strict",
    )
    bridge = adapter.adapt(dag)
    bridge_quality = ((bridge.model_config.get("meta") or {}).get("bridge") or {}).get("bridge_quality") or {}
    compatibility_quality = (bridge.init_metadata or {}).get("compatibility_quality") or {}
    source_nodes = (bridge.init_metadata or {}).get("source_nodes") or {}

    assert float(bridge_quality.get("effective_ops_ratio", 0.0)) > 0.0
    assert int(bridge_quality.get("effective_ops_count", 0)) > 0
    assert int(compatibility_quality.get("proxy_nodes_count", -1)) == 0
    assert int(compatibility_quality.get("unsupported_nodes_count", -1)) == 0
    assert int(compatibility_quality.get("aggregate_feature_nodes_count", 0)) >= 1
    assert "Mean" in ((bridge.model_config.get("model") or {}).get("features") or [])
    assert all(str(item.get("method") or "") != "mean" for item in source_nodes.values())


def test_rm101_strict_fails_fast_on_proxy():
    L = 128
    ch1 = _base_input(L)
    n_proxy = ProcessedData(
        node_id="sg_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="savgol_filter",
        results={},
        meta={"params": {"window_length": 11, "polyorder": 3}},
        shape=(1, L, 1),
    )
    dag = DAGState(
        user_instruction="rm101 strict proxy failfast",
        channels=["ch1"],
        nodes={"ch1": ch1, "sg_1": n_proxy},
        leaves=["sg_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=12800.0,
        max_layers=1,
        parallel_ops_per_layer=2,
        compat_profile="rm101_strict",
        enforce_tspn_closed_world=False,
    )
    with pytest.raises(ValueError, match="proxy operator is forbidden"):
        adapter.adapt(dag)


def test_rm101_strict_fails_fast_on_unsupported():
    L = 128
    ch1 = _base_input(L)
    n_bad = ProcessedData(
        node_id="cc_1",
        parents=["ch1", "ch1"],
        source_signal_id="ch1",
        method="cross_correlation",
        results={},
        meta={"params": {}},
        shape=(1, L, 1),
    )
    dag = DAGState(
        user_instruction="rm101 strict unsupported failfast",
        channels=["ch1"],
        nodes={"ch1": ch1, "cc_1": n_bad},
        leaves=["cc_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=12800.0,
        max_layers=1,
        parallel_ops_per_layer=2,
        unsupported_policy="drop",
        compat_profile="rm101_strict",
        enforce_tspn_closed_world=False,
    )
    with pytest.raises(ValueError, match="unsupported operator is forbidden"):
        adapter.adapt(dag)


def test_closed_world_contract_fails_fast_on_out_of_contract_operator():
    L = 128
    ch1 = _base_input(L)
    n_proxy = ProcessedData(
        node_id="sg_1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="savgol_filter",
        results={},
        meta={"params": {"window_length": 11, "polyorder": 3}},
        shape=(1, L, 1),
    )
    dag = DAGState(
        user_instruction="closed-world contract failfast",
        channels=["ch1"],
        nodes={"ch1": ch1, "sg_1": n_proxy},
        leaves=["sg_1"],
    )

    adapter = DAG2ConfigAdapter(
        in_dim=L,
        in_channels=1,
        num_classes=2,
        fs_hz=12800.0,
        max_layers=1,
        parallel_ops_per_layer=2,
        operator_contract="rm101_closed_v1",
        enforce_tspn_closed_world=True,
    )
    with pytest.raises(ValueError, match="contract violation"):
        adapter.adapt(dag)
