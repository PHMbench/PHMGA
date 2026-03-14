"""Minimal operator catalog used by the rebuilt research scaffold.

The catalog intentionally stays small, but it now exposes plan-facing operator
names so the NVTA-style planner can speak in compact `op_name` terms while the
backend still compiles against stable `op_uid` identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .base import BaseIsomorphicOperator, OperatorSpec


class NormalizeOperator(BaseIsomorphicOperator):
    """Per-channel normalization before feature extraction."""
    spec = OperatorSpec(
        op_uid="signal.normalize",
        name="Normalize",
        param_schema={"eps": "float"},
        input_shape_rule="CxT",
        output_shape_rule="CxT",
        backend_availability=["np", "sym"],
        execution_role="fixed",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        eps = float(kwargs.get("eps", 1e-6))
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + eps
        return (x - mean) / std

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"normalize({x_sym})"


class FFTMagnitudeOperator(BaseIsomorphicOperator):
    """Frequency-domain transform used by all current paths."""
    spec = OperatorSpec(
        op_uid="signal.fft_mag",
        name="FFT Magnitude",
        param_schema={},
        input_shape_rule="CxT",
        output_shape_rule="CxF",
        backend_availability=["np", "sym"],
        execution_role="fixed",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.abs(np.fft.rfft(x, axis=-1))

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"abs(rfft({x_sym}))"


class MeanFeatureOperator(BaseIsomorphicOperator):
    """Scalar summary feature on top of the transform chain."""
    spec = OperatorSpec(
        op_uid="feature.mean",
        name="Mean",
        param_schema={},
        input_shape_rule="CxT",
        output_shape_rule="1",
        backend_availability=["np", "sym"],
        execution_role="outer_only",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.asarray([float(x.mean())], dtype=float)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"mean({x_sym})"


class StdFeatureOperator(BaseIsomorphicOperator):
    """Standard-deviation summary feature for the minimal baseline."""
    spec = OperatorSpec(
        op_uid="feature.std",
        name="Std",
        param_schema={},
        input_shape_rule="CxT",
        output_shape_rule="1",
        backend_availability=["np", "sym"],
        execution_role="outer_only",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.asarray([float(x.std())], dtype=float)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"std({x_sym})"


class RMSFeatureOperator(BaseIsomorphicOperator):
    """Root-mean-square feature used in both downstream paths."""
    spec = OperatorSpec(
        op_uid="feature.rms",
        name="RMS",
        param_schema={},
        input_shape_rule="CxT",
        output_shape_rule="1",
        backend_availability=["np", "sym"],
        execution_role="outer_only",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.asarray([float(np.sqrt(np.mean(np.square(x))))], dtype=float)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"rms({x_sym})"


class ConcatenateFeatureOperator(BaseIsomorphicOperator):
    """Minimal multi-input feature fusion used to test NVTA-style execution."""

    spec = OperatorSpec(
        op_uid="multi.concatenate",
        name="Concatenate",
        param_schema={"axis": "int"},
        input_shape_rule="1 + 1 + ...",
        output_shape_rule="K",
        backend_availability=["np", "sym"],
        execution_role="proxy",
    )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        axis = int(kwargs.get("axis", 0))
        arrays = [np.asarray(part, dtype=float).reshape(-1) for part in x]
        return np.concatenate(arrays, axis=axis)

    def forward_sym(self, x_sym: str, **kwargs: float) -> str:
        return f"concat({x_sym})"


@dataclass
class OperatorCatalog:
    """Lookup table that exposes the sanctioned operator subset."""
    operators: Dict[str, BaseIsomorphicOperator]
    plan_name_aliases: Dict[str, str]

    def get(self, op_uid: str) -> BaseIsomorphicOperator:
        if op_uid not in self.operators:
            raise KeyError(f"Unknown operator: {op_uid}")
        return self.operators[op_uid]

    def resolve_plan_name(self, op_name: str) -> str:
        normalized = op_name.strip().lower()
        if normalized in self.plan_name_aliases:
            return self.plan_name_aliases[normalized]
        if normalized in self.operators:
            return normalized
        raise KeyError(f"Unknown plan operator name: {op_name}")

    def get_by_plan_name(self, op_name: str) -> BaseIsomorphicOperator:
        return self.get(self.resolve_plan_name(op_name))

    def specs(self) -> List[OperatorSpec]:
        return [operator.spec for operator in self.operators.values()]

    def feature_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.op_uid.startswith("feature.")]

    def transform_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.op_uid.startswith("signal.")]

    def summary(self) -> List[Dict[str, str]]:
        """Compact prompt-safe catalog summary."""

        summary: list[dict[str, str]] = []
        for spec in self.specs():
            summary.append(
                {
                    "op_uid": spec.op_uid,
                    "op_name": spec.op_uid.split(".")[-1],
                    "name": spec.name,
                    "input_shape_rule": spec.input_shape_rule,
                    "output_shape_rule": spec.output_shape_rule,
                    "execution_role": spec.execution_role,
                }
            )
        return summary


def get_operator_catalog() -> OperatorCatalog:
    """Build the small closed-world operator catalog used by tests and scripts."""
    operators: Iterable[BaseIsomorphicOperator] = (
        NormalizeOperator(),
        FFTMagnitudeOperator(),
        MeanFeatureOperator(),
        StdFeatureOperator(),
        RMSFeatureOperator(),
        ConcatenateFeatureOperator(),
    )
    plan_name_aliases = {
        "normalize": "signal.normalize",
        "fft": "signal.fft_mag",
        "fft_mag": "signal.fft_mag",
        "mean": "feature.mean",
        "std": "feature.std",
        "rms": "feature.rms",
        "concatenate": "multi.concatenate",
    }
    return OperatorCatalog(
        {operator.spec.op_uid: operator for operator in operators},
        plan_name_aliases=plan_name_aliases,
    )
