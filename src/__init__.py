"""Public API for the PHMGA package."""

# Avoid circular imports by using lazy imports
def get_llm():
    """Get configured LLM instance."""
    from .model import get_llm as _get_llm
    return _get_llm()

def build_builder_graph():
    """Build the builder graph."""
    from .phm_outer_graph import build_builder_graph as _build_builder_graph
    return _build_builder_graph()

def build_executor_graph():
    """Build the executor graph."""
    from .phm_outer_graph import build_executor_graph as _build_executor_graph
    return _build_executor_graph()

# Try to import optional graph module
try:
    from .graph import graph
except Exception:  # pragma: no cover - optional dependency may be missing
    graph = None

__all__ = ["build_builder_graph", "build_executor_graph", "graph", "get_llm"]
