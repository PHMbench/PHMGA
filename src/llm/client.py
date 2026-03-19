"""Compatibility shim for legacy imports.

The canonical implementations now live in:
- src.llm.base
- src.llm.structured
- src.llm.providers.*
- src.llm.factory
"""

from .providers import codex_cli as _codex_cli_module
from .base import LLMClient, LLMProviderError, LLMSchemaError
from .factory import get_llm
from .providers import CodexCliLLM, OfflineLLM, OpenAICodexLLM, OpenRouterLLM

# Re-export the concrete module objects used by CodexCliLLM so legacy
# monkeypatch targets like `src.llm.client.shutil.which` still affect the
# actual transport implementation after the refactor.
shutil = _codex_cli_module.shutil
subprocess = _codex_cli_module.subprocess

__all__ = [
    "CodexCliLLM",
    "LLMClient",
    "LLMProviderError",
    "LLMSchemaError",
    "OfflineLLM",
    "OpenAICodexLLM",
    "OpenRouterLLM",
    "shutil",
    "subprocess",
    "get_llm",
]
