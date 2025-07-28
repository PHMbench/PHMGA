from __future__ import annotations

import logging
from typing import Dict, Any

from ..configuration import Config
from ..state import PHMState
from ..prompts import PROMPT_REFLECTOR

__all__ = ["reflect"]


def reflect(state: PHMState, config: Config) -> Dict[str, Any]:
    """Evaluate whether analysis is sufficient."""
    logging.info("Reflecting on analysis results")
    iteration = state.get("iteration_count", 0) + 1
    sufficient = iteration >= config.max_loops or state.get("final_decision") == "OK"
    history = state.get("reflection_history", []) + [f"iteration {iteration}"]
    return {"is_sufficient": sufficient, "reflection_history": history, "iteration_count": iteration}
