from __future__ import annotations

from typing import Dict, Type

from .schemas import (
    PHMOperator,
    PatchOperator,
    CopyOperator,
    FFTOperator,
    MeanOperator,
    KurtosisOperator,
    SimilarityOperator,
)


_OPERATOR_REGISTRY: Dict[str, Type[PHMOperator]] = {
    "patch": PatchOperator,
    "copy": CopyOperator,
    "fft": FFTOperator,
    "mean": MeanOperator,
    "kurtosis": KurtosisOperator,
    "similarity": SimilarityOperator,
}


def create_operator(data: Dict) -> PHMOperator:
    op_name = data.get("op_name")
    if op_name not in _OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator {op_name}")
    cls = _OPERATOR_REGISTRY[op_name]
    return cls(**{k: v for k, v in data.items() if k != "op_name"})
