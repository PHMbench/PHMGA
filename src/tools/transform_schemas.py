# src/tools/signal_ops.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.fft
import scipy.stats

from typing import Literal

from .signal_processing_schemas import (
    register_op,
    ExpandOp,
    TransformOp,
    AggregateOp,
    DecisionOp,
)


class FFTOp(TransformOp):
    op_name = "fft"
    rank_class = "TRANSFORM"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        y = np.abs(np.fft.rfft(x, axis=-2))
        return y


@register_op
class NormalizeOp(TransformOp):
    """Normalize signal using z-score or min-max."""

    op_name = "normalize"
    description = "Normalize signal"

    method: Literal["z_score", "min_max"] = "z_score"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        if self.method == "z_score":
            mean = np.mean(x, axis=-2, keepdims=True)
            std = np.std(x, axis=-2, keepdims=True) + 1e-8
            return (x - mean) / std
        elif self.method == "min_max":
            min_val = np.min(x, axis=-2, keepdims=True)
            max_val = np.max(x, axis=-2, keepdims=True)
            return (x - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unsupported method {self.method}")


@register_op
class DetrendOp(TransformOp):
    """Remove polynomial trend from signal."""

    op_name = "detrend"
    description = "Remove trend from signal"

    type: Literal["linear", "constant"] = "linear"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return scipy.signal.detrend(x, type=self.type, axis=-2)


@register_op
class CepstrumOp(TransformOp):
    """Compute real cepstrum of the signal."""

    op_name = "cepstrum"
    description = "Cepstrum analysis"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        spectrum = np.fft.fft(x, axis=-2)
        log_spec = np.log(np.abs(spectrum) + 1e-8)
        cepstrum = np.fft.ifft(log_spec, axis=-2).real
        return cepstrum
