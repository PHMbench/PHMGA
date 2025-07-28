# src/tools/signal_ops.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.fft
import scipy.stats

from .signal_processing_schemas import register_op, ExpandOp, TransformOp, AggregateOp, DecisionOp
from utils.shape import assert_shape


# ---------- EXPAND 类 ---------- #
@register_op
class PatchOp(ExpandOp):
    op_name = "patch"
    description = "把 (B,L,C) 切成 (B,P,L',C)，窗口=patch_size, 步长=stride."

    patch_size: int = 256
    stride: int = 128

    def execute(self, x: npt.NDArray, **_) -> npt.NDArray:
        assert_shape(x, (None, None, None))
        B, L, C = x.shape
        P = 1 + max(0, (L - self.patch_size) // self.stride)
        patches = np.empty((B, P, self.patch_size, C), dtype=x.dtype)
        for i in range(P):
            s = i * self.stride
            patches[:, i] = x[:, s : s + self.patch_size, :]
        return patches