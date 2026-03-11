from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ConfigDict, Field


ALLOWED_LLM_PROVIDERS = {"openrouter"}
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = Field(default="openrouter")
    query_generator_model: str = Field(default="")
    phm_model: str = Field(default="")
    reflection_model: str = Field(default="")
    answer_model: str = Field(default="")
    fake_llm: bool = Field(default=False)
    openrouter_api_key: Optional[str] = Field(default=None)
    openrouter_base_url: Optional[str] = Field(default=None)
    openrouter_http_referer: Optional[str] = Field(default=None)
    openrouter_title: Optional[str] = Field(default=None)

    @classmethod
    def from_mapping(
        cls,
        llm_cfg: Mapping[str, Any] | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "OpenRouterLLMConfig":
        merged = bind_llm_env(llm_cfg, env=env)
        query_model = str(merged.get("QUERY_GENERATOR_MODEL") or merged.get("PHM_MODEL") or "").strip()
        payload = {
            "provider": str(merged.get("LLM_PROVIDER") or "openrouter").strip().lower(),
            "query_generator_model": query_model,
            "phm_model": str(merged.get("PHM_MODEL") or query_model).strip(),
            "reflection_model": str(merged.get("REFLECTION_MODEL") or query_model).strip(),
            "answer_model": str(merged.get("ANSWER_MODEL") or query_model).strip(),
            "fake_llm": str(merged.get("FAKE_LLM") or "").strip().lower() in {"1", "true", "yes", "y"},
            "openrouter_api_key": str(merged.get("OPENROUTER_API_KEY") or "").strip() or None,
            "openrouter_base_url": str(merged.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).strip() or None,
            "openrouter_http_referer": str(merged.get("OPENROUTER_HTTP_REFERER") or "").strip() or None,
            "openrouter_title": str(merged.get("OPENROUTER_TITLE") or "").strip() or None,
        }
        return cls.model_validate(payload)

    @classmethod
    def from_runnable_config(
        cls,
        config: RunnableConfig | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "OpenRouterLLMConfig":
        configurable = config["configurable"] if config and "configurable" in config else {}
        return cls.from_mapping(configurable, env=env)


def bind_llm_env(
    llm_cfg: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    merged = dict(os.environ)
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})

    cfg = dict(llm_cfg or {})
    if not cfg:
        return merged

    provider = str(cfg.get("provider") or merged.get("LLM_PROVIDER") or "openrouter").strip().lower()
    if provider:
        merged["LLM_PROVIDER"] = provider

    query_model = str(cfg.get("query_generator_model") or merged.get("QUERY_GENERATOR_MODEL") or "").strip()
    if query_model:
        merged["QUERY_GENERATOR_MODEL"] = query_model
        merged["PHM_MODEL"] = str(cfg.get("phm_model") or query_model).strip()
        merged["REFLECTION_MODEL"] = str(cfg.get("reflection_model") or query_model).strip()
        merged["ANSWER_MODEL"] = str(cfg.get("answer_model") or query_model).strip()

    if "fake_llm" in cfg:
        merged["FAKE_LLM"] = "true" if bool(cfg.get("fake_llm")) else "false"

    if cfg.get("openrouter_api_key"):
        merged["OPENROUTER_API_KEY"] = str(cfg["openrouter_api_key"]).strip()
    if cfg.get("openrouter_base_url"):
        merged["OPENROUTER_BASE_URL"] = str(cfg["openrouter_base_url"]).strip()
    elif not str(merged.get("OPENROUTER_BASE_URL") or "").strip():
        merged["OPENROUTER_BASE_URL"] = DEFAULT_OPENROUTER_BASE_URL
    if cfg.get("openrouter_http_referer"):
        merged["OPENROUTER_HTTP_REFERER"] = str(cfg["openrouter_http_referer"]).strip()
    if cfg.get("openrouter_title"):
        merged["OPENROUTER_TITLE"] = str(cfg["openrouter_title"]).strip()
    return merged


def normalize_llm_config(
    llm_cfg: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    cfg = OpenRouterLLMConfig.from_mapping(llm_cfg, env=env)
    return cfg.model_dump()


def validate_provider_env(
    env: Mapping[str, str] | None = None,
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    env_map = dict(os.environ)
    if env:
        env_map.update({str(k): str(v) for k, v in env.items()})

    provider = str(env_map.get("LLM_PROVIDER") or "openrouter").strip().lower()
    model = str(env_map.get("QUERY_GENERATOR_MODEL") or env_map.get("PHM_MODEL") or "").strip()
    errors: list[str] = []
    warnings: list[str] = []

    if provider not in ALLOWED_LLM_PROVIDERS:
        errors.append(
            f"Invalid llm.provider={provider!r}. Expected one of: {', '.join(sorted(ALLOWED_LLM_PROVIDERS))}."
        )
    if not model:
        errors.append("Missing QUERY_GENERATOR_MODEL (or PHM_MODEL).")
    if not str(env_map.get("OPENROUTER_API_KEY") or "").strip():
        errors.append("Missing OPENROUTER_API_KEY.")
    if not str(env_map.get("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).strip():
        errors.append("Missing OPENROUTER_BASE_URL.")

    report = {
        "provider": provider,
        "model": model,
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def validate_llm_config(
    llm_cfg: Mapping[str, Any] | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    normalized = normalize_llm_config(llm_cfg, env=env)
    provider = str(normalized.get("provider") or "").strip().lower()
    errors = []
    if provider not in ALLOWED_LLM_PROVIDERS:
        errors.append(
            f"Invalid llm.provider={provider!r}. Expected one of: {', '.join(sorted(ALLOWED_LLM_PROVIDERS))}."
        )
    if llm_cfg and not normalized["query_generator_model"]:
        errors.append("llm.query_generator_model is required when llm block is provided.")

    env_report = validate_provider_env(env=bind_llm_env(llm_cfg, env=env), strict=False)
    return {
        **normalized,
        "ok": len(errors) == 0 and bool(env_report.get("ok", False)),
        "errors": list(errors) + list(env_report.get("errors") or []),
        "warnings": list(env_report.get("warnings") or []),
        "source": "case_yaml" if llm_cfg else "env",
    }
