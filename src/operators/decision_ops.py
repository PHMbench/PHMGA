"""DECISION operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .base import BaseIsomorphicOperator, OperatorSpec
from .common import flatten_numeric, flatten_numeric_pt, require_torch


class ThresholdDecisionOperator(BaseIsomorphicOperator):
    """Auxiliary terminal decision node for report evidence."""

    spec = OperatorSpec(
        op_uid="decision.threshold",
        op_name="threshold",
        name="Threshold Decision",
        schema_category="DECISION",
        rank_class="terminal_decision",
        description="Convert a scalar or vector feature into a simple thresholded decision side-output.",
        input_spec={"arity": "single", "min_rank": 1, "semantic": "feature_or_signal_summary"},
        output_spec={"semantic": "decision_dict", "rank_behavior": "terminal"},
        param_schema={"threshold": "float"},
        param_defaults={"threshold": 0.5},
        param_docs={"threshold": "Decision threshold applied to the mean absolute score."},
        input_shape_rule="1|K|CxT",
        output_shape_rule="dict(score, decision)",
        backend_availability=["np", "pt", "sym"],
        execution_role="outer_only",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use as a terminal evidence node when the workflow wants an interpretable rule output.",
        llm_tunable_params=["threshold"],
    )

    def forward_np(self, x: Any, **kwargs: float) -> Dict[str, Any]:
        threshold = float(kwargs.get("threshold", 0.5))
        value = float(np.mean(np.abs(flatten_numeric(x))))
        return {
            "score": value,
            "decision": bool(value >= threshold),
            "threshold": threshold,
        }

    def forward_sym(self, x_sym: Any, **kwargs: float) -> str:
        return f"threshold({x_sym})"

    def forward_pt(self, x: Any, **kwargs: float) -> Dict[str, Any]:
        torch = require_torch()
        threshold = float(kwargs.get("threshold", 0.5))
        value = float(torch.mean(torch.abs(flatten_numeric_pt(x))).item())
        return {
            "score": value,
            "decision": bool(value >= threshold),
            "threshold": threshold,
        }


def get_decision_operators() -> Tuple[BaseIsomorphicOperator, ...]:
    """Return the runnable DECISION operators."""

    return (ThresholdDecisionOperator(),)
