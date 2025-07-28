"""Analysis agent."""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np

from ..state import PHMState, AnalysisInsight

__all__ = ["analyze"]

logger = logging.getLogger(__name__)


def analyze(state: PHMState) -> Dict[str, Any]:
    """Compare extracted features and generate insight."""
    if len(state["extracted_features"]) < 1:
        return {}
    ref_feats = state["extracted_features"][0].features
    test_feats = state["extracted_features"][-1].features
    ref_vec = np.array([[v for v in d.values()] for d in ref_feats])
    test_vec = np.array([[v for v in d.values()] for d in test_feats])
    dist = float(np.linalg.norm(ref_vec - test_vec))
    insight = AnalysisInsight(
        content=f"Similarity distance {dist:.3f}",
        severity_score=min(1.0, dist),
        supporting_feature_ids=[state["extracted_features"][-1].feature_set_id],
    )
    return {"analysis_results": [insight], "final_decision": "OK"}
