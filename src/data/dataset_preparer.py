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

from src.bridge import FeaturePipelinePlan, FeatureSpec, ModelBuildPlan
from src.operators import OperatorCatalog

from .protocol import SignalRecord


@dataclass
class DatasetView:
    """One split-level view of features, labels, and sample provenance."""

    X: np.ndarray
    y: np.ndarray
    sample_ids: List[str]


def _apply_spec(window: np.ndarray, spec: FeatureSpec, catalog: OperatorCatalog) -> float:
    """Apply one compiled feature spec to a single window."""

    current = window[[spec.channel_index], :]
    for op_uid in spec.transform_ops:
        current = catalog.get(op_uid).forward_np(current)
    feature_value = catalog.get(spec.feature_op).forward_np(current)
    return float(np.asarray(feature_value, dtype=float).reshape(-1)[0])


def build_dataset_views(
    plan: Union[FeaturePipelinePlan, ModelBuildPlan],
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
) -> Dict[str, DatasetView]:
    """Turn split-specific windows into dataset views for downstream paths."""

    outputs: Dict[str, DatasetView] = {}
    for split_name, records in split_records.items():
        features: list[list[float]] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        for record in records:
            for window in record.windows:
                features.append([_apply_spec(window, spec, catalog) for spec in plan.feature_specs])
                labels.append(record.label)
                sample_ids.append(record.sample_id)
        outputs[split_name] = DatasetView(
            X=np.asarray(features, dtype=float),
            y=np.asarray(labels, dtype=int),
            sample_ids=sample_ids,
        )
    return outputs
