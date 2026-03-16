from __future__ import annotations

import importlib
from pathlib import Path

from scripts.run_case import _run_frontend_loop
from src.bridge import compile_dag_for_path
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


def test_resolve_torch_device_supports_auto_cpu_and_cuda(monkeypatch):
    torch = _require_torch()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(_resolve_torch_device("auto")) == "cpu"
    assert str(_resolve_torch_device("cpu")) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(_resolve_torch_device("auto")) == "cuda"
    assert str(_resolve_torch_device("cuda:1")) == "cuda:1"
