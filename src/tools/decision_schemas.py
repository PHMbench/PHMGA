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


