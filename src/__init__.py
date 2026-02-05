"""Public API for the PHMGA package."""

try:
    from .graph import graph
except Exception:  # pragma: no cover - optional dependency may be missing
    graph = None

try:  # pragma: no cover - optional dependency may be missing (e.g., langgraph)
    # Decoupled graph builders
    from .phm_outer_graph import build_builder_graph, build_executor_graph
except Exception:
    build_builder_graph = None  # type: ignore
    build_executor_graph = None  # type: ignore

# LLM helper does not depend on langgraph
from .model import get_llm

# MODIFIED: Expose the new graph builders in the public API
__all__ = ["build_builder_graph", "build_executor_graph", "graph", "get_llm"]
