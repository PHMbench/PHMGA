# src/tools/signal_ops.py
from __future__ import annotations

from typing import Literal
import numpy as np
import numpy.typing as npt
import scipy.signal
from pydantic import Field

from .signal_processing_schemas import (
    register_op,
    TransformOp,
)


@register_op  # <-- 确保 FFTOp 也被注册
class FFTOp(TransformOp):
    op_name = "fft"
    description = "Computes the Fast Fourier Transform of a real-valued signal."

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        # rfft 用于实数输入，效率更高。np.abs() 获取幅值谱。
        y = np.abs(np.fft.rfft(x, axis=-2))
        return y


@register_op
class NormalizeOp(TransformOp):
    """Normalize signal using z-score or min-max."""

    op_name = "normalize"
    description = "Normalize signal using z-score or min-max scaling."
    method: Literal["z_score", "min_max"] = Field("z_score", description="The normalization method to use.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        if self.method == "z_score":
            mean = np.mean(x, axis=-2, keepdims=True)
            std = np.std(x, axis=-2, keepdims=True)
            return (x - mean) / (std + 1e-9)
        elif self.method == "min_max":
            min_val = np.min(x, axis=-2, keepdims=True)
            max_val = np.max(x, axis=-2, keepdims=True)
            range_val = max_val - min_val
            return (x - min_val) / (range_val + 1e-9)
        else:
            # Pydantic 的验证可以防止这种情况
            raise ValueError(f"Unknown normalization method: {self.method}")


@register_op
class DetrendOp(TransformOp):
    """Remove polynomial trend from signal."""

    op_name = "detrend"
    description = "Remove a linear or constant trend from the signal."
    type: Literal["linear", "constant"] = Field("linear", description="The type of trend to remove.")

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        return scipy.signal.detrend(x, type=self.type, axis=-2)


@register_op
class CepstrumOp(TransformOp):
    """Compute real cepstrum of the signal."""

    op_name = "cepstrum"
    description = "Computes the real cepstrum of the signal, useful for detecting harmonic structures."

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        if x.ndim < 1:
            return x
        
        spectrum = np.fft.fft(x, axis=-2)
        log_spec = np.log(np.abs(spectrum) + 1e-9) # 为防止 log(0) 增加极小值
        
        # 逆傅里叶变换得到倒谱
        cepstrum = np.fft.ifft(log_spec, axis=-2).real
        return cepstrum
