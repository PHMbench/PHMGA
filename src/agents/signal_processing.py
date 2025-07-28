from __future__ import annotations

import logging
from typing import Dict, Any, List

import numpy as np

from ..configuration import Config
from ..state import PHMState, ProcessedSignal
from ..utils import assert_shape

__all__ = ["process_signals"]


def _apply_method(data: np.ndarray, method: str) -> np.ndarray:
    """Placeholder processing method."""
    return data


def process_signals(state: PHMState, config: Config) -> Dict[str, Any]:
    """Process test signals according to the plan."""
    logging.info("Processing signals")
    raw = np.array(state["test_signal"].data)
    assert_shape(raw, (None, None, None))

    if config.use_patch:
        b, l, c = raw.shape
        p = l // config.patch_size
        patched = raw[:, : p * config.patch_size, :].reshape(
            b, p, config.patch_size, c
        )
        base = patched
    else:
        base = raw

    results = [_apply_method(base, m) for m in state["plan"]["processing_methods"]]

    processed: List[ProcessedSignal] = []
    for method, res in zip(state["plan"]["processing_methods"], results):
        processed.append(
            ProcessedSignal(
                source_signal_id=state["test_signal"].signal_id,
                method=method,
                processed_data=res.tolist(),
            )
        )

    return {"processed_signals": processed}
