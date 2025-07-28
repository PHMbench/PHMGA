from __future__ import annotations

from typing import ClassVar, Any, Literal

from pydantic import BaseModel, Field


class PHMOperator(BaseModel):
    """Base class for all PHM system operators."""

    op_name: str = Field(..., description="Unique operator name")
    op_type: Literal[
        "Dimension-Increasing",
        "Dimension-Identity",
        "Dimension-Reducing",
        "Decision",
    ]
    description: ClassVar[str] = ""

    def execute(self, data: Any, **kwargs) -> Any:  # pragma: no cover - abstract
        """Execute the operator on the provided data."""
        raise NotImplementedError


class PatchOperator(PHMOperator):
    op_name: str = "patch"
    op_type: Literal["Dimension-Increasing"] = "Dimension-Increasing"
    description: ClassVar[str] = (
        "Split a (B,L,C) signal into patches of shape (B,P,L',C)."
    )
    patch_size: int = 256
    stride: int = 128

    def execute(self, data: Any, **kwargs) -> Any:
        from .functions import patch_signal

        return patch_signal(data, self.patch_size, self.stride)


class CopyOperator(PHMOperator):
    op_name: str = "copy"
    op_type: Literal["Dimension-Increasing"] = "Dimension-Increasing"
    description: ClassVar[str] = (
        "Duplicate the input signal P times. Input (B,L,C) -> (B,P,L,C)."
    )
    num_copies: int = 2

    def execute(self, data: Any, **kwargs) -> Any:
        from .functions import copy_signal

        return copy_signal(data, self.num_copies)


class FFTOperator(PHMOperator):
    op_name: str = "fft"
    op_type: Literal["Dimension-Identity"] = "Dimension-Identity"
    description: ClassVar[str] = (
        "Apply FFT along the last dimension. (...,L,C) -> (...,F,C)."
    )

    def execute(self, data: Any, **kwargs) -> Any:
        from .functions import fft_signal

        return fft_signal(data)


class MeanOperator(PHMOperator):
    op_name: str = "mean"
    op_type: Literal["Dimension-Reducing"] = "Dimension-Reducing"
    description: ClassVar[str] = (
        "Compute mean along time axis. (B,L,C) -> (B,C)."
    )

    def execute(self, data: Any, **kwargs) -> Any:
        from .functions import mean_signal

        return mean_signal(data)


class KurtosisOperator(PHMOperator):
    op_name: str = "kurtosis"
    op_type: Literal["Dimension-Reducing"] = "Dimension-Reducing"
    description: ClassVar[str] = (
        "Compute kurtosis along time axis. (B,L,C) -> (B,C)."
    )

    def execute(self, data: Any, **kwargs) -> Any:
        from .functions import kurtosis_signal

        return kurtosis_signal(data)


class SimilarityOperator(PHMOperator):
    op_name: str = "similarity"
    op_type: Literal["Decision"] = "Decision"
    description: ClassVar[str] = (
        "Compare test features with reference features and return similarity score."
    )
    method: str = "cosine"
    threshold: float = 0.8

    def execute(self, data: Any, ref: Any | None = None, **kwargs) -> Any:
        from .functions import similarity

        if ref is None:
            raise ValueError("Reference data required for similarity computation")
        return similarity(data, ref, self.method, self.threshold)
