"""Utilities for language model instantiation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from langchain_community.chat_models import FakeListChatModel

from ..configuration import Configuration


_FAKE_LLM: FakeListChatModel | None = None


def _best_effort_load_dotenv() -> None:
    """Load environment variables from `.env` if possible.

    - Prefer python-dotenv when installed.
    - Fallback to a minimal parser to support environments without python-dotenv.
    - Never override already-set environment variables.
    """
    try:  # pragma: no cover
        from dotenv import load_dotenv

        # python-dotenv may assert in some non-file contexts (e.g., `python - <<'PY'`).
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",  # repo root
        ]
        for path in candidates:
            try:
                if path.exists() and load_dotenv(dotenv_path=path, override=False):
                    return
            except Exception:
                continue

        # Fallback: try the default search behavior (may work in file-based execution).
        try:
            load_dotenv(override=False)
        except Exception:
            pass
        return
    except Exception:
        pass

    # Minimal parser fallback
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # repo root
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
            return
        except Exception:
            continue


def get_llm(
    configurable: Optional[Configuration] = None,
    *,
    temperature: float = 1.0,
    max_retries: int = 2,
) -> Union["ChatGoogleGenerativeAI", FakeListChatModel]:
    """Return a chat model instance for agent use.

    Parameters
    ----------
    configurable : Optional[Configuration]
        Configuration object providing ``query_generator_model``. If ``None``, a
        new :class:`Configuration` will be created with environment variables.
    temperature : float, optional
        Sampling temperature for the model. Defaults to ``1.0``.
    max_retries : int, optional
        Maximum number of API retries. Defaults to ``2``.

    Returns
    -------
    Union[ChatGoogleGenerativeAI, FakeListChatModel]
        Instantiated LLM ready for calls. When running offline tests (or when
        ``FAKE_LLM=true``), returns a shared :class:`FakeListChatModel`.
    """
    if configurable is None:
        configurable = Configuration()

    # Best-effort .env loading (keeps the main codebase consistent with NVTA scripts).
    _best_effort_load_dotenv()

    env_fake = os.getenv("FAKE_LLM", "").strip().lower() in {"1", "true", "yes", "y"}
    fake_llm = bool(getattr(configurable, "fake_llm", False)) or env_fake

    if fake_llm:
        global _FAKE_LLM
        # Use a shared mock model for testing
        if _FAKE_LLM is None:
            responses = [
                '[{"op_name": "mean", "params": {"parent": "ch1"}}]',
                '{"decision": "finish", "reason": "analysis complete"}',
                '{"plan": []}',
            ]
            _FAKE_LLM = FakeListChatModel(responses=responses)
        return _FAKE_LLM

    provider = (
        os.getenv("LLM_PROVIDER")
        or getattr(configurable, "llm_provider", None)
        or "gemini"
    ).strip().lower()

    # --- OpenAI-compatible (GLM / DeepSeek / gateways) ---
    openai_base_url = (
        getattr(configurable, "openai_base_url", None)
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    openai_api_key = getattr(configurable, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")

    deepseek_base = (
        getattr(configurable, "deepseek_api_base", None)
        or os.getenv("DEEPSEEK_API_BASE")
    )
    deepseek_key = getattr(configurable, "deepseek_api_key", None) or os.getenv("DEEPSEEK_API_KEY")

    glm_base = getattr(configurable, "glm_api_base", None) or os.getenv("GLM_API_BASE")
    glm_key = getattr(configurable, "glm_api_key", None) or os.getenv("GLM_API_KEY")

    # Auto-detect if user provided OpenAI-compatible env vars but forgot to set provider.
    if provider == "auto":
        provider = "openai_compatible" if (openai_base_url or deepseek_base or glm_base) else "gemini"

    if provider in {"openai", "openai_compatible", "deepseek", "glm"} or openai_base_url or deepseek_base or glm_base:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "langchain_openai is required for OpenAI-compatible providers. "
                "Install it, or set LLM_PROVIDER=gemini."
            ) from e

        if provider == "deepseek":
            base_url = deepseek_base
            api_key = deepseek_key
        elif provider == "glm":
            base_url = glm_base or openai_base_url
            api_key = glm_key or openai_api_key
        else:
            base_url = openai_base_url or deepseek_base or glm_base
            api_key = openai_api_key or deepseek_key or glm_key

        if not api_key:
            raise ValueError(
                "Missing API key for OpenAI-compatible provider. "
                "Set OPENAI_API_KEY (or DEEPSEEK_API_KEY / GLM_API_KEY)."
            )
        if not base_url:
            raise ValueError(
                "Missing base_url for OpenAI-compatible provider. "
                "Set OPENAI_BASE_URL (or DEEPSEEK_API_BASE / GLM_API_BASE)."
            )

        # NOTE:
        # In some environments, `openai` and `httpx` versions may be mismatched and
        # `ChatOpenAI` may fail during client creation with:
        #   "Client.__init__() got an unexpected keyword argument 'proxies'"
        # Passing an explicit `http_client` avoids OpenAI's default httpx client
        # construction path that may use the deprecated `proxies=` kwarg.
        http_client = None
        http_async_client = None
        try:  # pragma: no cover
            import httpx  # type: ignore

            http_client = httpx.Client(timeout=30.0, follow_redirects=True, trust_env=False)
            http_async_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True, trust_env=False)
        except Exception:
            pass

        kwargs = {
            "model": configurable.query_generator_model,
            "temperature": temperature,
            "max_retries": max_retries,
            "api_key": api_key,
            "base_url": base_url,
        }
        if http_client is not None:
            kwargs["http_client"] = http_client
        if http_async_client is not None:
            kwargs["http_async_client"] = http_async_client

        return ChatOpenAI(**kwargs)

    # --- Gemini (optional dependency) ---
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "langchain_google_genai is required for real LLM calls. "
            "Install compatible versions of langchain/langchain_google_genai, "
            "or set FAKE_LLM=true to run offline."
        ) from e

    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=temperature,
        max_retries=max_retries,
        api_key=api_key,
    )
