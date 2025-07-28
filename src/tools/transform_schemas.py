# src/tools/signal_ops.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.fft
import scipy.stats

from .signal_processing_schemas import register_op, ExpandOp, TransformOp, AggregateOp, DecisionOp


class FFTOp(TransformOp):
    op_name = "fft"
    rank_class = "TRANSFORM"

    def execute(self, x: np.ndarray, **kw) -> np.ndarray:
        y = np.abs(np.fft.rfft(x, axis=-2))
        return y