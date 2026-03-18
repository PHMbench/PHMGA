"""Split-aware dataset views built from materialized signal windows.

This module absorbs the useful part of the old `dataset_preparer_agent`
without turning it back into a workflow agent. Its role is narrower: turn
window-level split records plus compiled feature specs into train/val/test
dataset views that downstream runners can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from src.bridge import CompiledExecutionNode, CompiledOutputSpec, FeaturePipelinePlan, ModelBuildPlan
from src.operators import OperatorCatalog

from .protocol import SignalRecord


@dataclass
class DatasetView:
    """One split-level view of features, labels, and sample provenance."""

    X: np.ndarray
    y: np.ndarray
    sample_ids: List[str]


@dataclass
class TorchDatasetView:
    """One split-level tensor view used by the graph-level torch path."""

    X: "torch.Tensor"
    y: "torch.Tensor"
    sample_ids: List[str]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for torch dataset-view materialization.")
    return torch


def _ordered_inputs_np(values_by_node: Dict[str, np.ndarray], node: CompiledExecutionNode) -> List[np.ndarray]:
    if node.input_bindings:
        binding_items = sorted(
            node.input_bindings.items(),
            key=lambda item: int(item[0].removeprefix("arg")),
        )
        return [values_by_node[parent_id] for _, parent_id in binding_items]
    return [values_by_node[parent_id] for parent_id in node.parents]


def _execute_compiled_plan_np(
    window: np.ndarray,
    execution_nodes: List[CompiledExecutionNode],
    output_specs: List[CompiledOutputSpec],
    catalog: OperatorCatalog,
) -> np.ndarray:
    """Execute one compiled subgraph for a single window with NumPy operators."""

    values_by_node: Dict[str, np.ndarray] = {}
    for node in execution_nodes:
        if node.kind == "input":
            if node.channel_index is None:
                raise ValueError(f"Input node {node.node_id} is missing channel_index.")
            values_by_node[node.node_id] = np.asarray(window[[node.channel_index], :], dtype=float)
            continue
        operator = catalog.get(node.op_uid)
        if node.kind == "multi":
            parent_values = _ordered_inputs_np(values_by_node, node)
            values_by_node[node.node_id] = np.asarray(operator.forward_np(parent_values, **node.params), dtype=float)
            continue
        if len(node.parents) != 1:
            raise ValueError(f"Node {node.node_id} expects exactly one parent, got {len(node.parents)}.")
        parent_value = values_by_node[node.parents[0]]
        values_by_node[node.node_id] = np.asarray(operator.forward_np(parent_value, **node.params), dtype=float)

    outputs = [np.asarray(values_by_node[spec.output_node_id], dtype=float).reshape(-1) for spec in output_specs]
    if not outputs:
        return np.zeros((0,), dtype=float)
    return np.concatenate(outputs, axis=0)


def _ordered_inputs_pt(
    values_by_node: Dict[str, "torch.Tensor"],
    node: CompiledExecutionNode,
) -> List["torch.Tensor"]:
    if node.input_bindings:
        binding_items = sorted(
            node.input_bindings.items(),
            key=lambda item: int(item[0].removeprefix("arg")),
        )
        return [values_by_node[parent_id] for _, parent_id in binding_items]
    return [values_by_node[parent_id] for parent_id in node.parents]


def _execute_compiled_plan_pt(
    window: np.ndarray,
    execution_nodes: List[CompiledExecutionNode],
    output_specs: List[CompiledOutputSpec],
    catalog: OperatorCatalog,
    *,
    device: "torch.device",
) -> "torch.Tensor":
    """Execute one compiled subgraph for a single window with tensor operators."""

    torch_module = _require_torch()
    values_by_node: Dict[str, "torch.Tensor"] = {}
    for node in execution_nodes:
        if node.kind == "input":
            if node.channel_index is None:
                raise ValueError(f"Input node {node.node_id} is missing channel_index.")
            values_by_node[node.node_id] = torch_module.as_tensor(
                window[[node.channel_index], :],
                dtype=torch_module.float32,
                device=device,
            )
            continue
        operator = catalog.get(node.op_uid)
        if node.kind == "multi":
            parent_values = _ordered_inputs_pt(values_by_node, node)
            result = operator.forward_pt(parent_values, **node.params)
        else:
            if len(node.parents) != 1:
                raise ValueError(f"Node {node.node_id} expects exactly one parent, got {len(node.parents)}.")
            result = operator.forward_pt(values_by_node[node.parents[0]], **node.params)
        if not torch_module.is_tensor(result):
            raise TypeError(f"Operator {node.op_uid} must return a tensor for torch execution.")
        values_by_node[node.node_id] = result.to(dtype=torch_module.float32)

    outputs = [values_by_node[spec.output_node_id].reshape(-1).to(dtype=torch_module.float32) for spec in output_specs]
    if not outputs:
        return torch_module.zeros((0,), dtype=torch_module.float32, device=device)
    return torch_module.cat(outputs, dim=0)


def build_dataset_views_np(
    plan: Union[FeaturePipelinePlan, ModelBuildPlan],
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
) -> Dict[str, DatasetView]:
    """Turn split-specific windows into NumPy dataset views for downstream paths."""

    outputs: Dict[str, DatasetView] = {}
    for split_name, records in split_records.items():
        features: list[list[float]] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        for record in records:
            features.append(
                _execute_compiled_plan_np(record.window, plan.execution_nodes, plan.output_specs, catalog).tolist()
            )
            labels.append(record.label)
            sample_ids.append(record.window_id)
        feature_dim = sum(
            int(np.prod(node.shape_inference["out"]))
            for node in plan.manifest.nodes
            if any(spec.output_node_id == node.node_id for spec in plan.output_specs)
        )
        outputs[split_name] = DatasetView(
            X=np.asarray(features, dtype=float).reshape(len(features), feature_dim) if features else np.zeros((0, feature_dim), dtype=float),
            y=np.asarray(labels, dtype=int),
            sample_ids=sample_ids,
        )
    return outputs


def build_dataset_views_pt(
    plan: Union[FeaturePipelinePlan, ModelBuildPlan],
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    device: "torch.device | str",
) -> Dict[str, TorchDatasetView]:
    """Turn split-specific windows into tensor dataset views for the torch path."""

    torch_module = _require_torch()
    resolved_device = torch_module.device(device)
    outputs: Dict[str, TorchDatasetView] = {}
    for split_name, records in split_records.items():
        features: list["torch.Tensor"] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        for record in records:
            features.append(
                _execute_compiled_plan_pt(
                    record.window,
                    plan.execution_nodes,
                    plan.output_specs,
                    catalog,
                    device=resolved_device,
                )
            )
            labels.append(record.label)
            sample_ids.append(record.window_id)
        feature_dim = sum(
            int(np.prod(node.shape_inference["out"]))
            for node in plan.manifest.nodes
            if any(spec.output_node_id == node.node_id for spec in plan.output_specs)
        )
        x_tensor = (
            torch_module.stack(features, dim=0)
            if features
            else torch_module.zeros((0, feature_dim), dtype=torch_module.float32, device=resolved_device)
        )
        y_tensor = torch_module.as_tensor(labels, dtype=torch_module.long, device=resolved_device)
        outputs[split_name] = TorchDatasetView(
            X=x_tensor,
            y=y_tensor,
            sample_ids=sample_ids,
        )
    return outputs


def build_dataset_views(
    plan: Union[FeaturePipelinePlan, ModelBuildPlan],
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
    *,
    backend: str = "np",
    device: "torch.device | str | None" = None,
) -> Dict[str, DatasetView] | Dict[str, TorchDatasetView]:
    """Compatibility wrapper that selects NumPy or torch dataset materialization."""

    normalized = backend.strip().lower()
    if normalized == "np":
        return build_dataset_views_np(plan, split_records, catalog)
    if normalized == "pt":
        return build_dataset_views_pt(plan, split_records, catalog, device=device or "cpu")
    raise ValueError(f"Unsupported dataset view backend: {backend}")
