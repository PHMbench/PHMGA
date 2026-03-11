from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_community.chat_models import FakeListChatModel

from src.config import OpenRouterLLMConfig


def best_effort_load_dotenv() -> None:
    try:  # pragma: no cover
        from dotenv import load_dotenv

        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[3] / ".env",
        ]
        for path in candidates:
            try:
                if path.exists() and load_dotenv(dotenv_path=path, override=False):
                    return
            except Exception:
                continue
        try:
            load_dotenv(override=False)
        except Exception:
            pass
        return
    except Exception:
        pass

    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"):
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text or text.startswith("#") or "=" not in text:
                    continue
                key, value = text.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return
        except Exception:
            continue


def build_fake_llm(shared_fake_llm=None) -> FakeListChatModel:
    if shared_fake_llm is not None:
        return shared_fake_llm
    return FakeListChatModel(
        responses=[
            '[{"op_name": "mean", "params": {"parent": "ch1"}}]',
            '{"decision": "finish", "reason": "analysis complete"}',
            '{"plan": []}',
        ]
    )


def build_openrouter_llm(
    config: OpenRouterLLMConfig,
    *,
    model_name: str,
    temperature: float = 1.0,
    max_retries: int = 2,
):
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "langchain_openai is required for OpenRouter-backed LLM calls."
        ) from exc

    api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
    base_url = config.openrouter_base_url or os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY.")
    if not base_url:
        raise ValueError("Missing OPENROUTER_BASE_URL.")

    headers = {}
    referer = config.openrouter_http_referer or os.getenv("OPENROUTER_HTTP_REFERER")
    title = config.openrouter_title or os.getenv("OPENROUTER_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_retries": max_retries,
        "api_key": api_key,
        "base_url": base_url,
    }
    if headers:
        kwargs["default_headers"] = headers
    return ChatOpenAI(**kwargs)


def resolve_role_model(
    config: OpenRouterLLMConfig,
    *,
    role: str = "query_generator",
    model_name: Optional[str] = None,
) -> str:
    if model_name:
        return str(model_name).strip()
    role_key = str(role or "query_generator").strip().lower()
    if role_key == "reflection":
        return config.reflection_model or config.query_generator_model
    if role_key in {"answer", "report"}:
        return config.answer_model or config.query_generator_model
    if role_key == "phm":
        return config.phm_model or config.query_generator_model
    return config.query_generator_model or config.phm_model
