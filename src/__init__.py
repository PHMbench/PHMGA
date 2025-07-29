"""Public API for the PHMGA package."""

try:
    from .graph import graph
except Exception:  # pragma: no cover - optional dependency may be missing
    graph = None

from .phm_outer_graph import build_outer_graph
from .model import get_llm

__all__ = ["build_outer_graph", "graph", "get_llm"]
