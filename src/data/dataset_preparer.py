"""Split-aware dataset views built from materialized signal windows.

This module absorbs the useful part of the old `dataset_preparer_agent`
without turning it back into a workflow agent. Its role is narrower: turn
split records plus compiled feature specs into train/val/test dataset views
that downstream runners can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None

from src.bridge import FeaturePipelinePlan, FeatureSpec, ModelBuildPlan
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


def _apply_spec_np(window: np.ndarray, spec: FeatureSpec, catalog: OperatorCatalog) -> float:
    """Apply one compiled feature spec to a single window with NumPy execution."""

    current = window[[spec.channel_index], :]
    for op_uid in spec.transform_ops:
        current = catalog.get(op_uid).forward_np(current)
    feature_value = catalog.get(spec.feature_op).forward_np(current)
    return float(np.asarray(feature_value, dtype=float).reshape(-1)[0])


def _apply_spec_pt(window: np.ndarray, spec: FeatureSpec, catalog: OperatorCatalog, *, device: "torch.device") -> "torch.Tensor":
    """Apply one compiled feature spec to a single window with tensor execution."""

    torch_module = _require_torch()
    current = torch_module.as_tensor(window[[spec.channel_index], :], dtype=torch_module.float32, device=device)
    for op_uid in spec.transform_ops:
        current = catalog.get(op_uid).forward_pt(current)
    feature_value = catalog.get(spec.feature_op).forward_pt(current)
    if not torch_module.is_tensor(feature_value):
        raise TypeError(f"Feature operator {spec.feature_op} must return a tensor for torch execution.")
    return feature_value.reshape(-1).to(dtype=torch_module.float32)


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
            for window in record.windows:
                features.append([_apply_spec_np(window, spec, catalog) for spec in plan.feature_specs])
                labels.append(record.label)
                sample_ids.append(record.sample_id)
        outputs[split_name] = DatasetView(
            X=np.asarray(features, dtype=float),
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
            for window in record.windows:
                parts = [_apply_spec_pt(window, spec, catalog, device=resolved_device) for spec in plan.feature_specs]
                features.append(torch_module.cat(parts, dim=0))
                labels.append(record.label)
                sample_ids.append(record.sample_id)
        feature_dim = len(plan.feature_specs)
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
