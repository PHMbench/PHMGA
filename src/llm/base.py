"""Base contracts for runtime LLM backends."""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    provider: str
    mode: str
    model: str
    api_key_env: str

    def generate_json(self, prompt: str, *, repair_prompt: str | None = None) -> Dict[str, Any]:
        ...

    def generate_text(self, prompt: str) -> str:
        ...


class LLMBackendError(RuntimeError):
    """Raised when an LLM provider transport or parsing step fails."""
