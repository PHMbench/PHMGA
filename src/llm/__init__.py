from __future__ import annotations

from typing import Optional

from src.config import OpenRouterLLMConfig

from .openrouter import best_effort_load_dotenv, build_fake_llm, build_openrouter_llm, resolve_role_model


def get_llm(
    configurable: Optional[OpenRouterLLMConfig] = None,
    *,
    role: str = "query_generator",
    model_name: str | None = None,
    temperature: float = 1.0,
    max_retries: int = 2,
    fake_llm_override=None,
):
    best_effort_load_dotenv()
    config = configurable or OpenRouterLLMConfig.from_runnable_config(None)
    if config.fake_llm:
        return build_fake_llm(fake_llm_override)
    resolved_model = resolve_role_model(config, role=role, model_name=model_name)
    return build_openrouter_llm(
        config,
        model_name=resolved_model,
        temperature=temperature,
        max_retries=max_retries,
    )


def get_default_llm(
    configurable: Optional[OpenRouterLLMConfig] = None,
    *,
    model_name: str | None = None,
    temperature: float = 1.0,
    max_retries: int = 2,
    fake_llm_override=None,
):
    return get_llm(
        configurable,
        role="query_generator",
        model_name=model_name,
        temperature=temperature,
        max_retries=max_retries,
        fake_llm_override=fake_llm_override,
    )


__all__ = ["OpenRouterLLMConfig", "get_default_llm", "get_llm"]
