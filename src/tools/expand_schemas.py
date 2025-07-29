# src/tools/signal_ops.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.fft
import scipy.stats

from .signal_processing_schemas import register_op, ExpandOp, TransformOp, AggregateOp, DecisionOp
from .utils import assert_shape
from numpy.lib.stride_tricks import as_strided


# ---------- EXPAND 类 ---------- #
@register_op
class PatchOp(ExpandOp):
    op_name = "patch"
    description = "Transforms input of shape [..., L, C] into patches of shape [..., P, L', C], where P is number of patches and L' is patch_size"

    patch_size: int = 256
    stride: int = 128

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        
        # Get shape info
        shape = x.shape
        L = shape[-2]  # Length dimension (second to last)
        
        # Calculate number of complete patches
        P_complete = max(0, (L - self.patch_size) // self.stride + 1)
        
        # Calculate total number of patches
        P = max(1, 1 + (L - self.patch_size) // self.stride)
        
        # Prepare output shape
        output_shape = shape[:-2] + (P, self.patch_size) + shape[-1:]
        result = np.zeros(output_shape, dtype=x.dtype)
        
        if P_complete > 0:
            # Create view shape and strides for complete patches
            view_shape = shape[:-2] + (P_complete, self.patch_size) + shape[-1:]
            
            # Calculate strides for the new view
            strides = x.strides
            view_strides = strides[:-2] + (self.stride * strides[-2],) + strides[-2:]
            
            # Create the view with stride tricks - this is fast and memory-efficient
            patches = as_strided(x, shape=view_shape, strides=view_strides)
            
            # Copy the complete patches to the result
            result[..., :P_complete, :, :] = patches
        
        # Handle the last partial patch if needed
        if P_complete < P:
            start_idx = P_complete * self.stride
            end_idx = L
            actual_size = end_idx - start_idx
            result[..., P_complete, :actual_size, :] = x[..., start_idx:end_idx, :]

        return result


@register_op
class ScalogramOp(ExpandOp):
    """Compute scalogram using continuous wavelet transform."""

    op_name = "scalogram"
    description = "Continuous wavelet scalogram"

    wavelet: str = "morl"
    scales: list[int] = [1, 2, 3]

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        import pywt

        coeffs, _ = pywt.cwt(x, self.scales, self.wavelet, axis=-2)
        return np.abs(coeffs)
