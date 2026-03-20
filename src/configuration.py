"""LangChain-style configuration facade for the PHM frontend runtime."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


def _default_api_key_env(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "codex_cli":
        return ""
    return "OPENROUTER_API_KEY"


def _default_base_url(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "https://api.openai.com/v1"
    if normalized == "codex_cli":
        return ""
    return "https://openrouter.ai/api/v1"


def _default_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"openai", "codex_cli"}:
        return "gpt-5.3-codex"
    return "z-ai/glm-4.5-air:free"


def _stage_b_active_model(llm_cfg: Dict[str, Any], provider: str) -> str:
    stage_b_cfg = llm_cfg.get("stage_b", {})
    if not isinstance(stage_b_cfg, dict):
        return ""
    normalized = provider.strip().lower()
    if normalized == "codex_cli":
        return str(stage_b_cfg.get("codex_active_model") or "")
    if normalized == "openrouter":
        return str(stage_b_cfg.get("openrouter_active_model") or "")
    return ""


def _normalize_provider_defaults(provider: str, *, model: Any, api_key_env: Any, base_url: Any) -> Dict[str, str]:
    normalized_provider = provider.strip().lower()
    default_model = _default_model(normalized_provider)
    default_api_key_env = _default_api_key_env(normalized_provider)
    default_base_url = _default_base_url(normalized_provider)
    legacy_provider = "openrouter" if normalized_provider in {"openai", "codex_cli"} else "openai"
    legacy_model = _default_model(legacy_provider)
    legacy_api_key_env = _default_api_key_env(legacy_provider)
    legacy_base_url = _default_base_url(legacy_provider)

    chosen_model = str(model or default_model)
    chosen_api_key_env = str(api_key_env or default_api_key_env)
    chosen_base_url = str(base_url or default_base_url)
    if chosen_model == legacy_model:
        chosen_model = default_model
    if chosen_api_key_env == legacy_api_key_env:
        chosen_api_key_env = default_api_key_env
    if chosen_base_url == legacy_base_url:
        chosen_base_url = default_base_url
    return {
        "model": chosen_model,
        "api_key_env": chosen_api_key_env,
        "base_url": chosen_base_url,
    }


class Configuration(BaseModel):
    """Minimal frontend configuration exposed to LangChain/LangGraph agents."""

    provider: str = Field(default="codex_cli")
    mode: str = Field(default="offline_stub")
    model: str = Field(default="gpt-5.3-codex")
    api_key_env: str = Field(default="")
    base_url: str = Field(default="")
    timeout_sec: float = Field(default=300.0)
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
        provider = str(values.get("provider", "codex_cli"))
        values.update(
            _normalize_provider_defaults(
                provider,
                model=values.get("model"),
                api_key_env=values.get("api_key_env"),
                base_url=values.get("base_url"),
            )
        )
        return cls(**values)

    @classmethod
    def from_runtime_config(cls, runtime_config: Dict[str, Any]) -> "Configuration":
        llm_cfg = dict(runtime_config.get("llm", {}))
        runtime = dict(runtime_config.get("runtime", {}))
        provider = str(llm_cfg.get("provider", "codex_cli"))
        stage_b_active_model = _stage_b_active_model(llm_cfg, provider)
        provider_defaults = _normalize_provider_defaults(
            provider,
            model=llm_cfg.get("model") or stage_b_active_model,
            api_key_env=llm_cfg.get("api_key_env"),
            base_url=llm_cfg.get("base_url"),
        )
        return cls(
            provider=provider,
            mode=str(llm_cfg.get("mode", "offline_stub")),
            model=provider_defaults["model"],
            api_key_env=provider_defaults["api_key_env"],
            base_url=provider_defaults["base_url"],
            timeout_sec=float(llm_cfg.get("timeout_sec", 300.0)),
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
