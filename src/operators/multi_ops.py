"""MULTI_VARIABLE operators inspired by the C_Agent tools taxonomy."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .base import BaseIsomorphicOperator, OperatorSpec
from .common import (
    concat_symbolic,
    ensure_float_tensor,
    flatten_numeric,
    flatten_numeric_pt,
    require_torch,
    scalar_feature,
    scalar_output_tensor,
    scalar_feature_pt,
)


class ConcatenateFeatureOperator(BaseIsomorphicOperator):
    """Feature fusion for multiple scalar or vector branches."""

    spec = OperatorSpec(
        op_uid="multi.concatenate",
        op_name="concatenate",
        name="Concatenate",
        schema_category="MULTI_VARIABLE",
        rank_class="multi_input",
        description="Fuse multiple scalar or vector features into one combined feature vector.",
        input_spec={"arity": "multi", "min_parents": 2, "semantic": "feature_vector"},
        output_spec={"semantic": "feature_vector", "rank_behavior": "merge"},
        param_schema={"axis": "int"},
        param_defaults={"axis": 0},
        param_docs={"axis": "Concatenation axis after each parent result has been flattened to a 1-D feature vector."},
        input_shape_rule="1 + 1 + ...",
        output_shape_rule="K",
        backend_availability=["np", "pt", "sym"],
        execution_role="proxy",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use to combine multiple aggregate branches into a richer feature vector.",
        llm_tunable_params=["axis"],
    )

    def forward_np(self, x: Any, **kwargs: float) -> np.ndarray:
        axis = int(kwargs.get("axis", 0))
        arrays = [np.asarray(part, dtype=float).reshape(-1) for part in x]
        return np.concatenate(arrays, axis=axis)

    def forward_sym(self, x_sym: Any, **kwargs: float) -> str:
        return concat_symbolic(x_sym, "concat")

    def forward_pt(self, x: Any, **kwargs: float):
        torch = require_torch()
        axis = int(kwargs.get("axis", 0))
        arrays = [
            ensure_float_tensor(part, op_name=self.spec.op_uid).reshape(-1)
            for part in x
        ]
        return torch.cat(arrays, dim=axis)


class CrossCorrelationOperator(BaseIsomorphicOperator):
    """Cross-correlation score between two or more parent branches."""

    spec = OperatorSpec(
        op_uid="multi.cross_correlation",
        op_name="cross_correlation",
        name="Cross Correlation",
        schema_category="MULTI_VARIABLE",
        rank_class="multi_input",
        description="Measure shared structure across parent branches with normalized cross-correlation.",
        input_spec={"arity": "multi", "min_parents": 2, "semantic": "aligned_numeric_branches"},
        output_spec={"semantic": "scalar_feature", "rank_behavior": "merge_reduce"},
        input_shape_rule="(CxT)+(CxT) or (1)+(1)",
        output_shape_rule="1",
        backend_availability=["np", "pt", "sym"],
        execution_role="proxy",
        legal_paths=["dag_only", "ml", "torch"],
        planning_notes="Use when the relationship between channels or branches is informative, not just their individual summaries.",
        llm_tunable_params=[],
    )

    def forward_np(self, x: Any, **kwargs: float) -> np.ndarray:
        del kwargs
        arrays = [flatten_numeric(part) for part in x]
        if len(arrays) < 2:
            return scalar_feature(0.0)
        scores: list[float] = []
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
                corr = np.correlate(left_norm, right_norm, mode="full")
                scores.append(float(np.max(np.abs(corr)) / length))
        return scalar_feature(float(np.mean(scores)) if scores else 0.0)

    def forward_sym(self, x_sym: Any, **kwargs: float) -> str:
        return concat_symbolic(x_sym, "cross_correlation")

    def forward_pt(self, x: Any, **kwargs: float):
        del kwargs
        torch = require_torch()
        arrays = [flatten_numeric_pt(part).to(dtype=torch.float64) for part in x]
        if len(arrays) < 2:
            return scalar_feature_pt(0.0)
        scores: list[Any] = []
        for index, left in enumerate(arrays):
            for right in arrays[index + 1 :]:
                length = min(left.numel(), right.numel())
                if length == 0:
                    scores.append(torch.tensor(0.0, dtype=left.dtype, device=left.device))
                    continue
                left_cut = left[:length]
                right_cut = right[:length]
                left_std = torch.std(left_cut, unbiased=False)
                right_std = torch.std(right_cut, unbiased=False)
                if float(left_std.item()) <= 1e-12 or float(right_std.item()) <= 1e-12:
                    scores.append(torch.tensor(0.0, dtype=left.dtype, device=left.device))
                    continue
                left_norm = (left_cut - torch.mean(left_cut)) / left_std
                right_norm = (right_cut - torch.mean(right_cut)) / right_std
                corr = torch.nn.functional.conv1d(
                    left_norm.view(1, 1, -1),
                    right_norm.view(1, 1, -1),
                    padding=length - 1,
                ).view(-1)
                scores.append(torch.max(torch.abs(corr)) / float(length))
        return scalar_output_tensor(torch.stack(scores).mean(), like=arrays[0]) if scores else scalar_feature_pt(0.0)


def get_multi_operators() -> Tuple[BaseIsomorphicOperator, ...]:
    """Return the runnable MULTI_VARIABLE operators."""

    return (
        ConcatenateFeatureOperator(),
        CrossCorrelationOperator(),
    )
