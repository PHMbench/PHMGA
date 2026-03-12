from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .base import BaseIsomorphicOperator, OperatorSpec


class NormalizeOperator(BaseIsomorphicOperator):
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


@dataclass
class OperatorCatalog:
    operators: Dict[str, BaseIsomorphicOperator]

    def get(self, op_uid: str) -> BaseIsomorphicOperator:
        if op_uid not in self.operators:
            raise KeyError(f"Unknown operator: {op_uid}")
        return self.operators[op_uid]

    def specs(self) -> List[OperatorSpec]:
        return [operator.spec for operator in self.operators.values()]

    def feature_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.op_uid.startswith("feature.")]

    def transform_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.op_uid.startswith("signal.")]


def get_operator_catalog() -> OperatorCatalog:
    operators: Iterable[BaseIsomorphicOperator] = (
        NormalizeOperator(),
        FFTMagnitudeOperator(),
        MeanFeatureOperator(),
        StdFeatureOperator(),
        RMSFeatureOperator(),
    )
    return OperatorCatalog({operator.spec.op_uid: operator for operator in operators})
