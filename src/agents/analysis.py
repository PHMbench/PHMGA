from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np

from ..state import PHMState, AnalysisInsight

__all__ = ["analyze"]


def analyze(state: PHMState, config) -> Dict[str, Any]:
    """Compare extracted features and produce an insight."""
    logging.info("Analyzing features")
    if not state["extracted_features"]:
        return {}

    ref = np.array(state["reference_signal"].data)
    ref_vec = ref.mean(axis=(1, 0)).mean(axis=0)

    test_feat = state["extracted_features"][-1].features
    test_vec = np.array([[v for v in row.values()] for row in test_feat]).mean(axis=0)

    diff = float(np.linalg.norm(test_vec - ref_vec))
    severity = float(1 / (1 + diff))
    insight = AnalysisInsight(
        content=f"difference {diff:.3f}",
        severity_score=severity,
        supporting_feature_ids=[state["extracted_features"][-1].feature_set_id],
    )
    decision = "ALERT" if severity < 0.5 else "OK"
    return {"analysis_results": state["analysis_results"] + [insight], "final_decision": decision}
