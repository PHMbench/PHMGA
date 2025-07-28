from __future__ import annotations

import logging
from typing import Dict, Any

from ..state import PHMState

__all__ = ["dispatcher"]


def dispatcher(state: PHMState) -> Dict[str, Any]:
    """Prepare for parallel processing branches."""
    logging.info("Dispatching tasks according to plan")
    return {}
