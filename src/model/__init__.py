"""Compatibility shim that routes old imports to the new LLM backends."""

from __future__ import annotations

from typing import Optional

from ..configuration import Configuration
from ..llm import get_llm as _get_runtime_llm


def get_llm(configurable: Optional[Configuration | dict] = None, **_: object):
    if configurable is None:
        return _get_runtime_llm({"llm": Configuration.from_runnable_config(None).model_dump()})
    if isinstance(configurable, Configuration):
        return _get_runtime_llm({"llm": configurable.model_dump()})
    return _get_runtime_llm(configurable)
