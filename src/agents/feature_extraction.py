from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np

from ..state import PHMState, ProcessedSignal, ExtractedFeatures

__all__ = ["extract_features"]


def _extract(data: np.ndarray, method: str) -> List[Dict[str, float]]:
    if data.ndim == 4:
        vec = data.mean(axis=(1, 2))
    else:
        vec = data.mean(axis=1)
    return [{f"f{i}": float(v) for i, v in enumerate(row)} for row in vec]


def extract_features(state: PHMState, config) -> Dict[str, Any]:
    """Extract features from processed signals."""
    logging.info("Extracting features")
    results = []
    for proc in state["processed_signals"]:
        data = np.array(proc.processed_data)
        results.append(_extract(data, "mean"))
    features: List[ExtractedFeatures] = []
    for proc, res in zip(state["processed_signals"], results):
        features.append(
            ExtractedFeatures(source_processed_id=proc.processed_id, features=res)
        )
    return {"extracted_features": features}
