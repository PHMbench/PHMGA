"""Public API for the PHMGA package."""

try:
    from .graph import graph
except Exception:  # pragma: no cover - optional dependency may be missing
    graph = None

try:
    from .phm_outer_graph import build_builder_graph, build_executor_graph
except Exception:  # pragma: no cover - optional dependency may be missing
    build_builder_graph = None
    build_executor_graph = None
from .model import get_llm
from .config import load_runtime_config
from .runtime import run_experiment, run_preflight

__all__ = [
    "build_builder_graph",
    "build_executor_graph",
    "graph",
    "get_llm",
    "load_runtime_config",
    "run_experiment",
    "run_preflight",
]
