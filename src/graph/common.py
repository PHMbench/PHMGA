from __future__ import annotations

import traceback
from typing import Any, Dict, Iterable

try:  # pragma: no cover
    from langgraph.graph import END, START, StateGraph  # type: ignore

    LANGGRAPH_OK = True
except Exception:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = "__END__"  # type: ignore
    START = "__START__"  # type: ignore
    LANGGRAPH_OK = False

from src.states.phm_states import PHMState
from src.utils.logging_setup import get_current_logger, log_event, timed


class FallbackGraph:
    def __init__(self, steps: Iterable[tuple[str, Any]]):
        self._steps = list(steps)

    def stream(self, state: PHMState, config: Any | None = None):
        for name, fn in self._steps:
            update = run_node(name, fn, state)
            if isinstance(update, dict):
                fields = getattr(state.__class__, "model_fields", {})
                for key, value in update.items():
                    if key in fields:
                        setattr(state, key, value)
            yield {name: update}


def summarize_update(update: Any) -> Dict[str, Any]:
    if not isinstance(update, dict):
        return {"update_type": type(update).__name__}
    summary: Dict[str, Any] = {"keys": sorted(update.keys())}
    lengths: Dict[str, int] = {}
    for key, value in update.items():
        if isinstance(value, (list, tuple, dict, set)):
            try:
                lengths[key] = len(value)
            except Exception:
                continue
    if lengths:
        summary["lengths"] = lengths
    return summary


def run_node(name: str, fn: Any, state: PHMState) -> Dict[str, Any]:
    logger = get_current_logger()
    with timed(logger, event="node", phase="graph", node=name, message="Node execution"):
        try:
            update = fn(state)
            log_event(
                logger,
                level="INFO",
                event="state_update",
                phase="graph",
                node=name,
                message="Node returned state update.",
                payload=summarize_update(update),
            )
            return update
        except Exception as exc:
            log_event(
                logger,
                level="ERROR",
                event="node.exception",
                phase="graph",
                node=name,
                message=str(exc),
                payload={"exc_type": type(exc).__name__, "traceback": traceback.format_exc()},
            )
            raise
