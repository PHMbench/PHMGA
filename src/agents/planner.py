from __future__ import annotations

import logging
from typing import Dict, Any

from ..configuration import Config
from ..state import PHMState
from ..prompts import PROMPT_PLANNER

__all__ = ["planner"]


def planner(state: PHMState, config: Config) -> Dict[str, Any]:
    """Create an initial analysis plan.

    This implementation uses configuration parameters to build a deterministic
    plan suitable for testing.
    """
    logging.info("Planning analysis workflow")
    plan = {
        "processing_methods": config.signal_processing_methods or ["identity"],
        "feature_methods": config.feature_methods or ["mean"],
        "similarity_method": config.similarity_method,
        "max_loops": config.max_loops,
    }
    return {"plan": plan}
