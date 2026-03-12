"""Feature extraction helpers shared by the ML and trainable paths."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.bridge import FeaturePipelinePlan, FeatureSpec, ModelBuildPlan
from src.data import SignalRecord
from src.operators import OperatorCatalog


def _apply_spec(window: np.ndarray, spec: FeatureSpec, catalog: OperatorCatalog) -> float:
    """Apply one compiled feature spec to a single window."""
    current = window[[spec.channel_index], :]
    for op_uid in spec.transform_ops:
        current = catalog.get(op_uid).forward_np(current)
    feature_value = catalog.get(spec.feature_op).forward_np(current)
    return float(feature_value[0])


def build_feature_matrix(
    plan: FeaturePipelinePlan | ModelBuildPlan,
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
) -> Dict[str, Dict[str, np.ndarray | list[str]]]:
    """Turn split-specific signal windows into downstream feature matrices."""
    outputs: Dict[str, Dict[str, np.ndarray | list[str]]] = {}
    for split_name, records in split_records.items():
        features: list[list[float]] = []
        labels: list[int] = []
        sample_ids: list[str] = []
        for record in records:
            for window in record.windows:
                # Sample ids are duplicated per window on purpose so reports can
                # still trace predictions back to their source sample.
                features.append([_apply_spec(window, spec, catalog) for spec in plan.feature_specs])
                labels.append(record.label)
                sample_ids.append(record.sample_id)
        outputs[split_name] = {
            "X": np.asarray(features, dtype=float),
            "y": np.asarray(labels, dtype=int),
            "sample_ids": sample_ids,
        }
    return outputs
