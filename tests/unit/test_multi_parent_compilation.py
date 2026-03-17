from __future__ import annotations

from pathlib import Path

from scripts.run_case import _run_frontend_loop
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_dataset_views_np, build_dataset_views_pt, build_protocol_from_config, materialize_split_signals
from src.llm.client import OfflineLLM
from src.operators import get_operator_catalog
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _compiled_inputs(output_policy: str):
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_torch.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    llm = OfflineLLM()
    state = WorkflowState(
        user_instruction="Build a trainable PHM baseline.",
        dataset_name=protocol.dataset_name,
        graph_path="torch",
        max_iterations=4,
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "TEST"},
    )
    state = _run_frontend_loop(state, protocol, llm, catalog, config)
    compiled = compile_dag_for_path(state.dag, "torch", output_policy=output_policy)
    split_records = materialize_split_signals(protocol)
    return compiled, split_records, catalog


def test_terminal_only_output_policy_prefers_terminal_feature_and_multi_nodes():
    compiled, _, _ = _compiled_inputs("terminal_only")
    output_ids = {spec.output_node_id for spec in compiled.output_specs}

    assert any(node_id.startswith("concatenate_") for node_id in output_ids)
    assert any(node_id.startswith("cross_correlation_") for node_id in output_ids)
    assert any(node_id.startswith("band_power_13_") for node_id in output_ids)
    assert all(not node_id.startswith("threshold_") for node_id in output_ids)
    assert all(not node_id.startswith("kurtosis_05_") for node_id in output_ids)
    assert all(not node_id.startswith("spectral_centroid_07_") for node_id in output_ids)
    assert all(not node_id.startswith("band_power_11_") for node_id in output_ids)


def test_include_intermediate_features_retains_feature_nodes_and_terminal_multi_nodes():
    compiled, _, _ = _compiled_inputs("include_intermediate_features")
    output_ids = {spec.output_node_id for spec in compiled.output_specs}

    assert any(node_id.startswith("concatenate_") for node_id in output_ids)
    assert any(node_id.startswith("cross_correlation_") for node_id in output_ids)
    assert any(node_id.startswith("kurtosis_05_") for node_id in output_ids)
    assert any(node_id.startswith("spectral_centroid_07_") for node_id in output_ids)
    assert any(node_id.startswith("band_power_11_") for node_id in output_ids)
    assert all(not node_id.startswith("threshold_") for node_id in output_ids)


def test_dataset_views_change_dimension_with_output_policy_but_keep_np_pt_parity():
    terminal_compiled, split_records, catalog = _compiled_inputs("terminal_only")
    intermediate_compiled, _, _ = _compiled_inputs("include_intermediate_features")

    terminal_np = build_dataset_views_np(terminal_compiled, split_records, catalog)
    terminal_pt = build_dataset_views_pt(terminal_compiled, split_records, catalog, device="cpu")
    intermediate_np = build_dataset_views_np(intermediate_compiled, split_records, catalog)
    intermediate_pt = build_dataset_views_pt(intermediate_compiled, split_records, catalog, device="cpu")

    assert terminal_np["train"].X.shape[1] < intermediate_np["train"].X.shape[1]
    assert tuple(terminal_pt["train"].X.shape) == terminal_np["train"].X.shape
    assert tuple(intermediate_pt["train"].X.shape) == intermediate_np["train"].X.shape
