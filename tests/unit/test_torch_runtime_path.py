from __future__ import annotations

import importlib
from pathlib import Path

from scripts.run_case import _run_frontend_loop
from src.bridge import (
    CompiledDagManifest,
    CompiledExecutionNode,
    CompiledOutputSpec,
    ManifestNode,
    ModelBuildPlan,
    compile_dag_for_path,
)
from src.config import load_runtime_config
from src.data import (
    build_dataset_views,
    build_dataset_views_np,
    build_dataset_views_pt,
    build_protocol_from_config,
    materialize_split_signals,
)
from src.llm.client import OfflineLLM
from src.operators import get_operator_catalog
from src.states import WorkflowState
from src.training.module_runtime import GraphModule
from src.training.runner import _resolve_torch_device, run_torch_pipeline


ROOT = Path(__file__).resolve().parents[2]


def _require_torch():
    return importlib.import_module("torch")


def _compiled_torch_inputs():
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
    compiled = compile_dag_for_path(state.dag, "torch")
    split_records = materialize_split_signals(protocol)
    return config, protocol, catalog, compiled, split_records


def _compiled_multi_inputs():
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
    compiled = compile_dag_for_path(state.dag, "torch", output_policy="terminal_only")
    split_records = materialize_split_signals(protocol)
    return config, protocol, catalog, compiled, split_records


def test_build_dataset_views_pt_matches_np_shapes():
    torch = _require_torch()
    _, _, catalog, compiled, split_records = _compiled_torch_inputs()
    np_views = build_dataset_views_np(compiled, split_records, catalog)
    pt_views = build_dataset_views_pt(compiled, split_records, catalog, device="cpu")
    compat_views = build_dataset_views(compiled, split_records, catalog, backend="pt", device="cpu")

    assert set(np_views) == set(pt_views) == set(compat_views) == {"train", "val", "test"}
    assert torch.is_tensor(pt_views["train"].X)
    assert torch.is_tensor(pt_views["train"].y)
    assert tuple(pt_views["train"].X.shape) == np_views["train"].X.shape
    assert tuple(pt_views["val"].X.shape) == np_views["val"].X.shape
    assert tuple(pt_views["test"].X.shape) == np_views["test"].X.shape
    assert tuple(compat_views["train"].X.shape) == np_views["train"].X.shape


def test_run_torch_pipeline_uses_tensor_runtime_on_cpu():
    torch = _require_torch()
    config, _, catalog, compiled, split_records = _compiled_torch_inputs()
    artifacts = run_torch_pipeline(
        compiled,
        split_records,
        catalog,
        epochs=int(config["model"]["torch"]["epochs"]),
        learning_rate=float(config["model"]["torch"]["learning_rate"]),
        device="cpu",
    )

    assert artifacts["runtime_backend"] == "torch_tensor_runtime"
    assert len(artifacts["training_curves"]) == int(config["model"]["torch"]["epochs"])
    assert set(artifacts["checkpoint"]) == {"weight", "bias", "device"}
    assert artifacts["checkpoint"]["device"] == "cpu"
    assert set(artifacts["metrics"]) == {"train", "val", "test"}
    assert artifacts["predictions"]["test"]
    assert artifacts["similarity_artifacts"]["split_sizes"]["train"] > 0
    assert isinstance(artifacts["checkpoint"]["weight"], list)
    assert torch.tensor(artifacts["checkpoint"]["weight"]).ndim == 2


def test_run_torch_pipeline_supports_module_runtime_on_cpu():
    config, _, catalog, compiled, split_records = _compiled_torch_inputs()
    artifacts = run_torch_pipeline(
        compiled,
        split_records,
        catalog,
        epochs=1,
        learning_rate=float(config["model"]["torch"]["learning_rate"]),
        device="cpu",
        phase="module_runtime",
        module_runtime_enabled=True,
    )

    assert artifacts["runtime_backend"] == "torch_module_runtime"
    assert set(artifacts["checkpoint"]) == {"graph_module_state", "head_state", "device", "runtime_config"}
    assert artifacts["checkpoint"]["runtime_config"]["phase"] == "module_runtime"
    assert artifacts["metrics"]["test"]
    assert artifacts["similarity_artifacts"]["split_sizes"]["train"] > 0


def test_learnable_control_emits_single_and_multi_attention_stats():
    config, _, catalog, compiled, split_records = _compiled_multi_inputs()
    artifacts = run_torch_pipeline(
        compiled,
        split_records,
        catalog,
        epochs=1,
        learning_rate=float(config["model"]["torch"]["learning_rate"]),
        device="cpu",
        phase="learnable_control",
        control_default_mode="attention",
        tau=0.8,
        attention_heads=2,
        attention_dropout=0.0,
    )

    assert artifacts["runtime_backend"] == "torch_learnable_runtime"
    stats = artifacts["control_statistics"]
    assert stats
    assert any(summary["mode"] == "channel_self_attention" for summary in stats.values())
    assert any(summary["mode"] == "attention_fusion" for summary in stats.values())


def test_graph_module_wavefilters_support_learnable_params():
    torch = _require_torch()
    catalog = get_operator_catalog()
    plan = ModelBuildPlan(
        backend_target="torch",
        execution_nodes=[
            CompiledExecutionNode(
                node_id="input_0",
                op_uid="signal.input",
                kind="input",
                parents=[],
                params={"channel_index": 0},
                channel_index=0,
            ),
            CompiledExecutionNode(
                node_id="wave_1",
                op_uid="signal.wavefilters",
                kind="transform",
                parents=["input_0"],
                params={"center_ratio": 0.15, "bandwidth_ratio": 0.08},
            ),
        ],
        output_specs=[CompiledOutputSpec(output_node_id="wave_1", output_kind="feature")],
        output_policy="terminal_only",
        trainable_head={"input_dim": 128, "hidden_dim": 8, "output_dim": 2},
        manifest=CompiledDagManifest(
            dag_hash="wavefilters-test",
            topo_order=["input_0", "wave_1"],
            path_type="torch",
            nodes=[
                ManifestNode(
                    node_id="input_0",
                    op_uid="signal.input",
                    kind="input",
                    operator_category="INPUT",
                    rank_class="rank_same",
                    legal_paths=["dag_only", "ml", "torch"],
                    backend_availability=["np", "pt", "sym"],
                    shape_inference={"in": [1, 128], "out": [1, 128]},
                ),
                ManifestNode(
                    node_id="wave_1",
                    op_uid="signal.wavefilters",
                    kind="transform",
                    operator_category="TRANSFORM",
                    rank_class="rank_same",
                    legal_paths=["dag_only", "ml", "torch"],
                    backend_availability=["np", "pt", "sym"],
                    shape_inference={"in": [1, 128], "out": [1, 128]},
                ),
            ],
        ),
    )
    graph = GraphModule(
        plan,
        catalog,
        phase="learnable_control",
        control_default_mode="gated",
        tau=1.0,
        attention_heads=1,
        attention_dropout=0.0,
    )
    features = graph(torch.randn(2, 2, 128, dtype=torch.float32))
    stats = graph.control_statistics()

    assert tuple(features.shape) == (2, 128)
    assert "wave_1" in stats
    assert "learnable_params" in stats["wave_1"]
    assert {"center_ratio", "bandwidth_ratio"} <= set(stats["wave_1"]["learnable_params"])


def test_resolve_torch_device_supports_auto_cpu_and_cuda(monkeypatch):
    torch = _require_torch()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(_resolve_torch_device("auto")) == "cpu"
    assert str(_resolve_torch_device("cpu")) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(_resolve_torch_device("auto")) == "cuda"
    assert str(_resolve_torch_device("cuda:1")) == "cuda:1"
