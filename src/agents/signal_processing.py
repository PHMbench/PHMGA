"""Signal processing agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from ..configuration import Config
from ..state import ProcessedSignal, PHMState
from ..utils import assert_shape

__all__ = ["signal_processor"]

logger = logging.getLogger(__name__)


def _make_patches(data: np.ndarray, patch_size: int) -> np.ndarray:
    B, L, C = data.shape
    P = L // patch_size
    trimmed = data[:, : P * patch_size, :]
    patched = trimmed.reshape(B, P, patch_size, C)
    return patched


def signal_processor(state: PHMState, config: Config) -> Dict[str, Any]:
    """Process signals optionally with patching."""
    arr = np.array(state["test_signal"].data)
    assert_shape(arr, (None, None, None))

    if config.use_patch:
        arr = _make_patches(arr, config.patch_size)
        method = "patch"
    else:
        method = "identity"
    processed = ProcessedSignal(
        source_signal_id=state["test_signal"].signal_id,
        method=method,
        processed_data=arr.tolist(),
    )
    return {"processed_signals": [processed]}
