"""Split-aware dataset views built from materialized signal windows.

This module absorbs the useful part of the old `dataset_preparer_agent`
without turning it back into a workflow agent. Its role is narrower: turn
window-level split records plus compiled feature specs into train/val/test
dataset views that downstream runners can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Union

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


@dataclass
class DatasetExecutionOptions:
    """Execution-time controls used only by dataset-level evidence materialization."""

    mode: str = "default"
    enable_runtime_trace: bool = False
    dataset_name: str = "unknown"
    graph_path: str = "unknown"
    evidence_path: str = "unknown"
    sample_budget: Optional[int] = None
    cross_correlation_large_input_threshold: int = 2048
    cross_correlation_max_lag: int = 256


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


def _shape_list(value: np.ndarray) -> List[int]:
    return [int(dimension) for dimension in np.asarray(value).shape]


def _trace_payload_from_options(options: DatasetExecutionOptions) -> Dict[str, Any]:
    return {
        "dataset": options.dataset_name,
        "graph_path": options.graph_path,
        "evidence_path": options.evidence_path,
        "subset_size_per_split": options.sample_budget,
        "splits": [],
        "total_elapsed_ms": 0.0,
    }


def _bounded_cross_correlation_scalar(
    parent_values: List[np.ndarray],
    *,
    max_lag: int,
) -> np.ndarray:
    """Dataset-evidence-only approximation for expensive cross correlation."""

    arrays = [np.asarray(part, dtype=float).reshape(-1) for part in parent_values]
    if len(arrays) < 2:
        return np.asarray([0.0], dtype=float)
    scores: List[float] = []
    for index, left in enumerate(arrays):
        for right in arrays[index + 1 :]:
            length = min(left.size, right.size)
            if length == 0:
                scores.append(0.0)
                continue
            left_cut = left[:length]
            right_cut = right[:length]
            left_std = float(np.std(left_cut))
            right_std = float(np.std(right_cut))
            if left_std <= 1e-12 or right_std <= 1e-12:
                scores.append(0.0)
                continue
            left_norm = (left_cut - left_cut.mean()) / left_std
            right_norm = (right_cut - right_cut.mean()) / right_std
            bounded_lag = min(int(max_lag), max(length - 1, 0))
            padded = np.pad(right_norm, (bounded_lag, bounded_lag), mode="constant")
            windows = np.lib.stride_tricks.sliding_window_view(padded, length)
            dots = np.abs(windows @ left_norm)
            lags = bounded_lag - np.arange(windows.shape[0], dtype=int)
            overlap_lengths = length - np.abs(lags)
            best = float(np.max(dots / overlap_lengths))
            scores.append(best)
    return np.asarray([float(np.mean(scores)) if scores else 0.0], dtype=float)


def _execute_compiled_plan_np(
    window: np.ndarray,
    execution_nodes: List[CompiledExecutionNode],
    output_specs: List[CompiledOutputSpec],
    catalog: OperatorCatalog,
    *,
    execution_options: Optional[DatasetExecutionOptions] = None,
    split_name: Optional[str] = None,
    window_index: Optional[int] = None,
) -> tuple[np.ndarray, List[Dict[str, Any]], float]:
    """Execute one compiled subgraph for a single window with NumPy operators."""

    values_by_node: Dict[str, np.ndarray] = {}
    runtime_trace_entries: List[Dict[str, Any]] = []
    trace_enabled = bool(execution_options and execution_options.enable_runtime_trace)
    total_elapsed_ms = 0.0
    for node in execution_nodes:
        if node.kind == "input":
            if node.channel_index is None:
                raise ValueError(f"Input node {node.node_id} is missing channel_index.")
            values_by_node[node.node_id] = np.asarray(window[[node.channel_index], :], dtype=float)
            continue
        operator = catalog.get(node.op_uid)
        approximation = {
            "enabled": False,
            "reason": None,
            "max_lag": None,
        }
        started_at = perf_counter()
        if node.kind == "multi":
            parent_values = _ordered_inputs_np(values_by_node, node)
            if (
                execution_options is not None
                and execution_options.mode == "dataset_evidence"
                and node.op_uid == "multi.cross_correlation"
            ):
                flattened_length = max((int(np.asarray(parent).size) for parent in parent_values), default=0)
                if flattened_length > execution_options.cross_correlation_large_input_threshold:
                    approximation = {
                        "enabled": True,
                        "reason": "bounded_lag_for_large_inputs",
                        "max_lag": int(execution_options.cross_correlation_max_lag),
                    }
                    result = _bounded_cross_correlation_scalar(
                        parent_values,
                        max_lag=execution_options.cross_correlation_max_lag,
                    )
                else:
                    result = np.asarray(operator.forward_np(parent_values, **node.params), dtype=float)
            else:
                result = np.asarray(operator.forward_np(parent_values, **node.params), dtype=float)
        else:
            if len(node.parents) != 1:
                raise ValueError(f"Node {node.node_id} expects exactly one parent, got {len(node.parents)}.")
            parent_value = values_by_node[node.parents[0]]
            parent_values = [parent_value]
            result = np.asarray(operator.forward_np(parent_value, **node.params), dtype=float)
        elapsed_ms = float((perf_counter() - started_at) * 1000.0)
        total_elapsed_ms += elapsed_ms
        values_by_node[node.node_id] = result
        if trace_enabled and execution_options is not None:
            runtime_trace_entries.append(
                {
                    "split": split_name,
                    "node_id": node.node_id,
                    "op_uid": node.op_uid,
                    "operator": node.op_uid.split(".")[-1],
                    "input_shape": [_shape_list(parent) for parent in parent_values],
                    "output_shape": _shape_list(result),
                    "parent_count": len(parent_values),
                    "sample_budget": execution_options.sample_budget,
                    "window_index": window_index,
                    "elapsed_ms": elapsed_ms,
                    "mode": execution_options.mode,
                    "approximation": approximation,
                }
            )

    outputs = [np.asarray(values_by_node[spec.output_node_id], dtype=float).reshape(-1) for spec in output_specs]
    if not outputs:
        return np.zeros((0,), dtype=float), runtime_trace_entries, total_elapsed_ms
    return np.concatenate(outputs, axis=0), runtime_trace_entries, total_elapsed_ms


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
    *,
    execution_options: Optional[DatasetExecutionOptions] = None,
    runtime_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, DatasetView]:
    """Turn split-specific windows into NumPy dataset views for downstream paths."""

    outputs: Dict[str, DatasetView] = {}
    trace_payload = (
        _trace_payload_from_options(execution_options)
        if execution_options is not None and execution_options.enable_runtime_trace
        else None
    )
    total_elapsed_ms = 0.0
    for split_name, records in split_records.items():
        features: list[list[float]] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        split_nodes: List[Dict[str, Any]] = []
        split_elapsed_ms = 0.0
        for record in records:
            feature_vector, trace_entries, elapsed_ms = _execute_compiled_plan_np(
                record.window,
                plan.execution_nodes,
                plan.output_specs,
                catalog,
                execution_options=execution_options,
                split_name=split_name,
                window_index=len(sample_ids),
            )
            features.append(feature_vector.tolist())
            if trace_payload is not None:
                split_nodes.extend(trace_entries)
                split_elapsed_ms += elapsed_ms
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
        if trace_payload is not None:
            trace_payload["splits"].append(
                {
                    "split": split_name,
                    "window_count": len(records),
                    "nodes": split_nodes,
                    "total_elapsed_ms": float(split_elapsed_ms),
                }
            )
            total_elapsed_ms += split_elapsed_ms
    if trace_payload is not None:
        trace_payload["total_elapsed_ms"] = float(total_elapsed_ms)
        if runtime_trace is not None:
            runtime_trace.clear()
            runtime_trace.update(trace_payload)
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
    execution_options: Optional[DatasetExecutionOptions] = None,
    runtime_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, DatasetView] | Dict[str, TorchDatasetView]:
    """Compatibility wrapper that selects NumPy or torch dataset materialization."""

    normalized = backend.strip().lower()
    if normalized == "np":
        return build_dataset_views_np(
            plan,
            split_records,
            catalog,
            execution_options=execution_options,
            runtime_trace=runtime_trace,
        )
    if normalized == "pt":
        return build_dataset_views_pt(plan, split_records, catalog, device=device or "cpu")
    raise ValueError(f"Unsupported dataset view backend: {backend}")
