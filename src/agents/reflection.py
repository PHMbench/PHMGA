"""Reflection agent."""

from __future__ import annotations

import logging
from typing import Dict, Any

from ..prompts import PROMPT_REFLECTOR
from ..state import PHMState

__all__ = ["reflect"]

logger = logging.getLogger(__name__)


def reflect(state: PHMState) -> Dict[str, Any]:
    """Decide whether the analysis is sufficient."""
    logger.info("Reflecting on analysis")
    sufficient = state["iteration_count"] >= 1 or bool(state["analysis_results"])
    history = list(state["reflection_history"])
    history.append("checked")
    return {"is_sufficient": sufficient, "reflection_history": history}
