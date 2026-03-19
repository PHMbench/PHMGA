from __future__ import annotations

import numpy as np

from src.bridge import (
    CompiledDagManifest,
    CompiledExecutionNode,
    CompiledOutputSpec,
    FeaturePipelinePlan,
    ManifestNode,
)
from src.data.dataset_preparer import DatasetExecutionOptions, build_dataset_views_np
from src.data.protocol import SignalRecord
from src.operators import get_operator_catalog


def _cross_correlation_plan(signal_length: int) -> FeaturePipelinePlan:
    return FeaturePipelinePlan(
        execution_nodes=[
            CompiledExecutionNode(
                node_id="ch1",
                op_uid="input.channel",
                kind="input",
                parents=[],
                channel_index=0,
            ),
            CompiledExecutionNode(
                node_id="ch2",
                op_uid="input.channel",
                kind="input",
                parents=[],
                channel_index=1,
            ),
            CompiledExecutionNode(
                node_id="cross_correlation_01_ch1_ch2",
                op_uid="multi.cross_correlation",
                kind="multi",
                parents=["ch1", "ch2"],
            ),
        ],
        output_specs=[
            CompiledOutputSpec(
                output_node_id="cross_correlation_01_ch1_ch2",
                output_kind="multi",
            )
        ],
        output_policy="terminal_only",
        manifest=CompiledDagManifest(
            dag_hash="unit-test-hash",
            topo_order=["ch1", "ch2", "cross_correlation_01_ch1_ch2"],
            path_type="ml",
            nodes=[
                ManifestNode(
                    node_id="ch1",
                    op_uid="input.channel",
                    kind="input",
                    operator_category="INPUT",
                    rank_class="source",
                    legal_paths=["dag_only", "ml", "torch"],
                    backend_availability=["np", "pt"],
                    shape_inference={"in": [1, signal_length], "out": [1, signal_length]},
                ),
                ManifestNode(
                    node_id="ch2",
                    op_uid="input.channel",
                    kind="input",
                    operator_category="INPUT",
                    rank_class="source",
                    legal_paths=["dag_only", "ml", "torch"],
                    backend_availability=["np", "pt"],
                    shape_inference={"in": [1, signal_length], "out": [1, signal_length]},
                ),
                ManifestNode(
                    node_id="cross_correlation_01_ch1_ch2",
                    op_uid="multi.cross_correlation",
                    kind="multi",
                    operator_category="MULTI_VARIABLE",
                    rank_class="multi_input",
                    legal_paths=["dag_only", "ml", "torch"],
                    backend_availability=["np", "pt"],
                    shape_inference={"in": [1, signal_length], "out": [1]},
                ),
            ],
        ),
    )


def _record_from_channels(left: np.ndarray, right: np.ndarray) -> SignalRecord:
    return SignalRecord(
        source_sample_id="sample_01",
        window_index=0,
        window_id="sample_01_w0",
        split="train",
        label=0,
        window=np.vstack([left, right]),
    )


def test_cross_correlation_evidence_mode_matches_full_path_for_small_inputs():
    catalog = get_operator_catalog()
    left = np.linspace(-1.0, 1.0, 128, dtype=float)
    right = np.roll(left, 7)
    plan = _cross_correlation_plan(signal_length=left.size)
    trace: dict[str, object] = {}

    views = build_dataset_views_np(
        plan,
        {"train": [_record_from_channels(left, right)]},
        catalog,
        execution_options=DatasetExecutionOptions(
            mode="dataset_evidence",
            enable_runtime_trace=True,
            dataset_name="UNIT",
            graph_path="ml",
            evidence_path="ml",
            sample_budget=1,
        ),
        runtime_trace=trace,
    )

    expected = catalog.get("multi.cross_correlation").forward_np(
        [left.reshape(1, -1), right.reshape(1, -1)]
    ).reshape(-1)
    assert np.allclose(views["train"].X[0], expected)
    node_trace = trace["splits"][0]["nodes"][0]
    assert node_trace["approximation"]["enabled"] is False
    assert node_trace["op_uid"] == "multi.cross_correlation"


def test_cross_correlation_evidence_mode_uses_bounded_lag_for_large_inputs():
    catalog = get_operator_catalog()
    time_axis = np.linspace(0.0, 20.0 * np.pi, 4096, dtype=float)
    left = np.sin(time_axis)
    right = np.roll(left, 128) + 0.01 * np.cos(time_axis)
    plan = _cross_correlation_plan(signal_length=left.size)
    trace: dict[str, object] = {}

    views = build_dataset_views_np(
        plan,
        {"train": [_record_from_channels(left, right)]},
        catalog,
        execution_options=DatasetExecutionOptions(
            mode="dataset_evidence",
            enable_runtime_trace=True,
            dataset_name="UNIT",
            graph_path="ml",
            evidence_path="ml",
            sample_budget=1,
            cross_correlation_large_input_threshold=2048,
            cross_correlation_max_lag=256,
        ),
        runtime_trace=trace,
    )

    assert views["train"].X.shape == (1, 1)
    assert np.isfinite(views["train"].X).all()
    node_trace = trace["splits"][0]["nodes"][0]
    assert node_trace["approximation"]["enabled"] is True
    assert node_trace["approximation"]["reason"] == "bounded_lag_for_large_inputs"
    assert node_trace["approximation"]["max_lag"] == 256
    assert node_trace["mode"] == "dataset_evidence"
    assert node_trace["window_index"] == 0
    assert node_trace["parent_count"] == 2
    assert trace["total_elapsed_ms"] >= node_trace["elapsed_ms"]
