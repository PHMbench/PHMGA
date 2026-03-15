"""Compatibility wrapper for feature-matrix construction.

The canonical dataset assembly logic now lives in `src.data.dataset_preparer`.
This module keeps the old `build_feature_matrix()` entrypoint so existing tests
and runners can migrate incrementally.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from src.bridge import FeaturePipelinePlan, ModelBuildPlan
from src.data import SignalRecord, build_dataset_views
from src.operators import OperatorCatalog


def build_feature_matrix(
    plan: Union[FeaturePipelinePlan, ModelBuildPlan],
    split_records: Dict[str, List[SignalRecord]],
    catalog: OperatorCatalog,
) -> Dict[str, Dict[str, Union[np.ndarray, List[str]]]]:
    """Return backward-compatible dict payloads built from dataset views."""

    views = build_dataset_views(plan, split_records, catalog)
    return {
        split_name: {
            "X": view.X,
            "y": view.y,
            "sample_ids": view.sample_ids,
        }
        for split_name, view in views.items()
    }
