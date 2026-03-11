from __future__ import annotations

from typing import Any, Callable, Dict, List

from .builder_loop_graph import build_builder_graph
from .executor_tspn_graph import build_executor_graph
from .research_graph import build_research_graph


GraphBuilder = Callable[[], Any]

_REGISTRY: Dict[str, GraphBuilder] = {
    "builder_loop": build_builder_graph,
    "executor_tspn": build_executor_graph,
    "research_graph": build_research_graph,
}


def register_graph_builder(name: str, builder: GraphBuilder) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("graph builder name cannot be empty")
    _REGISTRY[key] = builder


def get_graph_builder(name: str) -> GraphBuilder:
    key = str(name).strip()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown graph builder: {key}")
    return _REGISTRY[key]


def build_selected_graph(name: str) -> Any:
    return get_graph_builder(name)()


def list_graph_builders() -> List[str]:
    return sorted(_REGISTRY)
