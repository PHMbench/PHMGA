from __future__ import annotations

"""Legacy compatibility facade for graph builders.

New logic lives under ``src.graph``. This module intentionally exposes the old
function names without carrying separate workflow definitions.
"""

from src.graph import build_builder_graph, build_executor_graph, build_outer_graph
from src.graph.common import LANGGRAPH_OK as _LANGGRAPH_OK

__all__ = [
    "_LANGGRAPH_OK",
    "build_builder_graph",
    "build_executor_graph",
    "build_outer_graph",
]
