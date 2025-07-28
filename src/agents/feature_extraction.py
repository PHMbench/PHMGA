"""Feature extraction agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from ..state import ExtractedFeatures, ProcessedSignal, PHMState
from ..utils import assert_shape

__all__ = ["feature_extractor"]

logger = logging.getLogger(__name__)


def _extract_mean(features: np.ndarray) -> List[Dict[str, float]]:
    B = features.shape[0]
    if features.ndim == 4:
        # (B,P,L',C) -> average over P and L'
        vec = features.mean(axis=(1, 2))
    else:
        vec = features.mean(axis=1)
    return [{f"ch{i}": float(v) for i, v in enumerate(sample)} for sample in vec]


def feature_extractor(state: PHMState) -> Dict[str, Any]:
    """Extract simple statistical features."""
    if not state["processed_signals"]:
        return {}
    proc = state["processed_signals"][-1]
    arr = np.array(proc.processed_data)
    feats = _extract_mean(arr)
    extracted = ExtractedFeatures(
        source_processed_id=proc.processed_id,
        features=feats,
    )
    return {"extracted_features": [extracted]}
