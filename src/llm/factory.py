"""Factory and default resolution for PHM LLM clients."""

from __future__ import annotations

from typing import Any, Dict

from .base import LLMClient
from .env import load_runtime_dotenv
from .providers import BigModelLLM, CodexCliLLM, OfflineLLM, OpenAICodexLLM, OpenRouterLLM


def _default_api_key_env(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "OPENAI_API_KEY"
    if normalized == "codex_cli":
        return ""
    if normalized == "bigmodel":
        return "BIGMODEL_API_KEY"
    return "OPENROUTER_API_KEY"


def _default_base_url(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return "https://api.openai.com/v1"
    if normalized == "codex_cli":
        return ""
    if normalized == "bigmodel":
        return "https://open.bigmodel.cn/api/paas/v4"
    return "https://openrouter.ai/api/v1"


def _default_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized in {"openai", "codex_cli"}:
        return "gpt-5.3-codex"
    if normalized == "bigmodel":
        return "glm-4.7-flash"
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
    if normalized == "bigmodel":
        return str(stage_b_cfg.get("bigmodel_active_model") or "")
    return ""


def get_llm(config: Dict[str, Any]) -> LLMClient:
    """Resolve the configured LLM client."""

    llm_cfg = dict(config.get("llm", {}))
    load_runtime_dotenv(llm_cfg.get("env_file"))
    provider = str(llm_cfg.get("provider", "codex_cli"))
    mode = str(llm_cfg.get("mode", "offline_stub"))
    default_api_key_env = _default_api_key_env(provider)
    default_base_url = _default_base_url(provider)
    default_model = _default_model(provider)
    stage_b_active_model = _stage_b_active_model(llm_cfg, provider)
    legacy_provider = "openrouter" if provider in {"openai", "codex_cli"} else "openai"
    legacy_api_key_env = _default_api_key_env(legacy_provider)
    legacy_base_url = _default_base_url(legacy_provider)
    legacy_model = _default_model(legacy_provider)

    configured_api_key_env = str(llm_cfg.get("api_key_env") or default_api_key_env)
    configured_base_url = str(llm_cfg.get("base_url") or default_base_url)
    configured_model = str(llm_cfg.get("model") or stage_b_active_model or default_model)
    if configured_api_key_env == legacy_api_key_env:
        configured_api_key_env = default_api_key_env
    if configured_base_url == legacy_base_url:
        configured_base_url = default_base_url
    if configured_model == legacy_model and not llm_cfg.get("model") and not stage_b_active_model:
        configured_model = default_model

    if mode == "offline_stub":
        return OfflineLLM(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "offline-stub")),
            api_key_env=configured_api_key_env,
        )
    if mode == "provider" and provider == "openrouter":
        return OpenRouterLLM(
            provider=provider,
            mode=mode,
            model=configured_model,
            api_key_env=configured_api_key_env,
            base_url=configured_base_url,
            timeout_sec=float(llm_cfg.get("timeout_sec", 30.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            retry_once=bool(config.get("runtime", {}).get("provider_retry_once", True)),
            retry_backoff_sec=float(config.get("runtime", {}).get("provider_retry_backoff_sec", 2.0)),
            retry_max_backoff_sec=float(config.get("runtime", {}).get("provider_retry_max_backoff_sec", 10.0)),
            http_referer=str(llm_cfg.get("http_referer", "")).strip() or None,
            app_title=str(llm_cfg.get("app_title", "PHMGA")).strip() or None,
        )
    if mode == "provider" and provider == "openai":
        return OpenAICodexLLM(
            provider=provider,
            mode=mode,
            model=configured_model,
            api_key_env=configured_api_key_env,
            base_url=configured_base_url,
            timeout_sec=float(llm_cfg.get("timeout_sec", 30.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            retry_once=bool(config.get("runtime", {}).get("provider_retry_once", True)),
            retry_backoff_sec=float(config.get("runtime", {}).get("provider_retry_backoff_sec", 2.0)),
            retry_max_backoff_sec=float(config.get("runtime", {}).get("provider_retry_max_backoff_sec", 10.0)),
            http_referer=str(llm_cfg.get("http_referer", "")).strip() or None,
            app_title=str(llm_cfg.get("app_title", "")).strip() or None,
        )
    if mode == "provider" and provider == "bigmodel":
        return BigModelLLM(
            provider=provider,
            mode=mode,
            model=configured_model,
            api_key_env=configured_api_key_env,
            base_url=configured_base_url,
            timeout_sec=float(llm_cfg.get("timeout_sec", 30.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            retry_once=bool(config.get("runtime", {}).get("provider_retry_once", True)),
            retry_backoff_sec=float(config.get("runtime", {}).get("provider_retry_backoff_sec", 2.0)),
            retry_max_backoff_sec=float(config.get("runtime", {}).get("provider_retry_max_backoff_sec", 10.0)),
            http_referer=str(llm_cfg.get("http_referer", "")).strip() or None,
            app_title=str(llm_cfg.get("app_title", "")).strip() or None,
        )
    if mode == "provider" and provider == "codex_cli":
        return CodexCliLLM(
            provider=provider,
            mode=mode,
            model=configured_model,
            api_key_env=configured_api_key_env,
            base_url=configured_base_url,
            timeout_sec=float(llm_cfg.get("timeout_sec", 300.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            retry_once=bool(config.get("runtime", {}).get("provider_retry_once", True)),
            http_referer=str(llm_cfg.get("http_referer", "")).strip() or None,
            app_title=str(llm_cfg.get("app_title", "")).strip() or None,
            planner_smoke_timeout_sec=float(llm_cfg.get("planner_smoke_timeout_sec", 15.0)),
        )
    raise ValueError(f"Unsupported llm configuration: provider={provider}, mode={mode}, model={configured_model}")


__all__ = ["get_llm"]
