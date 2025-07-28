"""Dispatcher agent."""

from __future__ import annotations

from typing import Dict, Any

from ..state import PHMState

__all__ = ["dispatcher"]


def dispatcher(state: PHMState) -> Dict[str, Any]:
    """Prepare tasks for parallel processing.

    In this simplified implementation it just returns an empty update.
    """
    return {}
