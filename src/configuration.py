"""Compatibility configuration facade for the simplified runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    provider: str = Field(default="openrouter")
    mode: str = Field(default="offline_stub")
    model: str = Field(default="z-ai/glm-4.5-air:free")
    api_key_env: str = Field(default="OPENROUTER_API_KEY")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    timeout_sec: float = Field(default=60.0)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2000)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        raw_values: Dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }
        values = {key: value for key, value in raw_values.items() if value is not None}
        return cls(**values)

    @classmethod
    def from_runtime_config(cls, runtime_config: Dict[str, Any]) -> "Configuration":
        llm_cfg = dict(runtime_config.get("llm", {}))
        return cls(
            provider=str(llm_cfg.get("provider", "openrouter")),
            mode=str(llm_cfg.get("mode", "offline_stub")),
            model=str(llm_cfg.get("model", "z-ai/glm-4.5-air:free")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 60.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", 2000)),
        )
