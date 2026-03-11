"""Model package public API.

This module no longer contains provider-specific LLM construction logic.
LLM runtime is delegated to ``src.llm`` and kept here only for compatibility.
"""

from __future__ import annotations

from src.llm import get_default_llm as _get_default_llm
from src.llm import get_llm as _get_llm

_FAKE_LLM = None


def get_llm(configurable=None, *, role: str = "query_generator", model_name=None, temperature: float = 1.0, max_retries: int = 2):
    return _get_llm(
        configurable,
        role=role,
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        fake_llm_override=_FAKE_LLM,
    )


def get_default_llm(configurable=None, model_name=None, **kwargs):
    return _get_default_llm(
        configurable,
        model_name=model_name,
        fake_llm_override=_FAKE_LLM,
        **kwargs,
    )


__all__ = ["_FAKE_LLM", "get_default_llm", "get_llm"]
