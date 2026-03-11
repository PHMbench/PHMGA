from __future__ import annotations

from typing import Any, Callable


def forward_agent_call(agent_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return agent_fn(*args, **kwargs)
