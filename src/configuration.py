"""LangChain-style configuration facade for the PHM frontend runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Minimal frontend configuration exposed to LangChain/LangGraph agents."""

    provider: str = Field(default="openrouter")
    mode: str = Field(default="offline_stub")
    model: str = Field(default="stepfun/step-3.5-flash:free")
    api_key_env: str = Field(default="OPENROUTER_API_KEY")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    timeout_sec: float = Field(default=30.0)
    temperature: float = Field(default=0.0)
    max_tokens_structured: int = Field(default=800)
    max_tokens_report: int = Field(default=2000)
    http_referer: Optional[str] = Field(default=None)
    app_title: Optional[str] = Field(default="PHMGA")
    provider_retry_once: bool = Field(default=True)

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
        runtime = dict(runtime_config.get("runtime", {}))
        return cls(
            provider=str(llm_cfg.get("provider", "openrouter")),
            mode=str(llm_cfg.get("mode", "offline_stub")),
            model=str(llm_cfg.get("model", "stepfun/step-3.5-flash:free")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 30.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            http_referer=llm_cfg.get("http_referer"),
            app_title=llm_cfg.get("app_title", "PHMGA"),
            provider_retry_once=bool(runtime.get("provider_retry_once", True)),
        )

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "llm": {
                "provider": self.provider,
                "mode": self.mode,
                "model": self.model,
                "api_key_env": self.api_key_env,
                "base_url": self.base_url,
                "timeout_sec": self.timeout_sec,
                "temperature": self.temperature,
                "max_tokens_structured": self.max_tokens_structured,
                "max_tokens_report": self.max_tokens_report,
                "http_referer": self.http_referer,
                "app_title": self.app_title,
            },
            "runtime": {
                "provider_retry_once": self.provider_retry_once,
            },
        }
