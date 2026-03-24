"""Minimal OpenRouter, Gemini, and offline backends."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict

import httpx

from .base import LLMBackendError


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise LLMBackendError("No JSON object found in provider response.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"Provider response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMBackendError("Provider returned JSON but not an object.")
    return payload


@dataclass
class OfflineBackend:
    provider: str = "offline"
    mode: str = "offline_stub"
    model: str = "offline-stub"
    api_key_env: str = ""

    def generate_json(self, prompt: str, *, repair_prompt: str | None = None) -> Dict[str, Any]:
        del repair_prompt
        try:
            return _extract_json_object(prompt)
        except Exception:
            return {}

    def generate_text(self, prompt: str) -> str:
        return prompt


@dataclass
class OpenRouterBackend:
    provider: str = "openrouter"
    mode: str = "provider"
    model: str = "z-ai/glm-4.5-air:free"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 2000
    max_retries: int = 3

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        backoff = [2.0, 5.0, 10.0]
        index = max(0, min(attempt - 1, len(backoff) - 1))
        return backoff[index]

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise LLMBackendError(f"Missing API key in env var {self.api_key_env}.")
        return api_key

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_sec),
            write=10.0,
            pool=5.0,
        )

    def _request(self, prompt: str, *, expect_json: bool) -> str:
        del expect_json
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        timeout = self._timeout()
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = client.post(
                            f"{self.base_url.rstrip('/')}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self._api_key()}",
                                "Content-Type": "application/json",
                            },
                            json=body,
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 429 and attempt < self.max_retries:
                            time.sleep(self._retry_delay(attempt, exc.response.headers.get("Retry-After")))
                            continue
                        raise
        except httpx.ConnectTimeout as exc:
            raise LLMBackendError(
                f"OpenRouter connect timeout after {timeout.connect:.1f}s: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMBackendError(
                f"OpenRouter read timeout after {timeout.read:.1f}s while waiting for the response body: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMBackendError(f"OpenRouter transport failed: {exc}") from exc
        payload = response.json()
        try:
            return str(payload["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            raise LLMBackendError(f"OpenRouter returned unexpected payload: {payload}") from exc

    def generate_json(self, prompt: str, *, repair_prompt: str | None = None) -> Dict[str, Any]:
        text = self._request(prompt, expect_json=True)
        try:
            return _extract_json_object(text)
        except LLMBackendError:
            retry_prompt = repair_prompt or (
                "Return only a valid JSON object with no prose.\n\nOriginal task:\n" + prompt
            )
            repaired = self._request(retry_prompt, expect_json=True)
            return _extract_json_object(repaired)

    def generate_text(self, prompt: str) -> str:
        return self._request(prompt, expect_json=False)


@dataclass
class BigModelBackend:
    provider: str = "bigmodel"
    mode: str = "provider"
    model: str = "glm-4.7-flashx"
    api_key_env: str = "BIGMODEL_API_KEY"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 2000
    max_retries: int = 3
    thinking_type: str = "disabled"
    clear_thinking: bool | None = None

    def _retry_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        backoff = [2.0, 5.0, 10.0]
        index = max(0, min(attempt - 1, len(backoff) - 1))
        return backoff[index]

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise LLMBackendError(f"Missing API key in env var {self.api_key_env}.")
        return api_key

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=10.0,
            read=float(self.timeout_sec),
            write=10.0,
            pool=5.0,
        )

    def _request(self, prompt: str, *, expect_json: bool) -> str:
        del expect_json
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        thinking_type = str(self.thinking_type or "").strip().lower()
        if thinking_type:
            body["thinking"] = {"type": thinking_type}
            if self.clear_thinking is not None:
                body["thinking"]["clear_thinking"] = bool(self.clear_thinking)
        timeout = self._timeout()
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                for attempt in range(1, self.max_retries + 1):
                    try:
                        response = client.post(
                            f"{self.base_url.rstrip('/')}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self._api_key()}",
                                "Content-Type": "application/json",
                            },
                            json=body,
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as exc:
                        if exc.response.status_code == 429 and attempt < self.max_retries:
                            time.sleep(self._retry_delay(attempt, exc.response.headers.get("Retry-After")))
                            continue
                        raise
        except httpx.ConnectTimeout as exc:
            raise LLMBackendError(
                f"BigModel connect timeout after {timeout.connect:.1f}s: {exc}"
            ) from exc
        except httpx.ReadTimeout as exc:
            raise LLMBackendError(
                f"BigModel read timeout after {timeout.read:.1f}s while waiting for the response body: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMBackendError(f"BigModel transport failed: {exc}") from exc
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
            content = message.get("content")
            if isinstance(content, list):
                text_parts = [
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                content = "".join(text_parts)
            content_text = str(content or "")
            if content_text.strip():
                return content_text
            reasoning_text = str(message.get("reasoning_content") or "").strip()
            if reasoning_text:
                raise LLMBackendError(
                    "BigModel returned reasoning_content but empty content. "
                    "Disable thinking with llm.thinking_type=disabled for non-reasoning text generation."
                )
            return content_text
        except Exception as exc:
            raise LLMBackendError(f"BigModel returned unexpected payload: {payload}") from exc

    def generate_json(self, prompt: str, *, repair_prompt: str | None = None) -> Dict[str, Any]:
        text = self._request(prompt, expect_json=True)
        try:
            return _extract_json_object(text)
        except LLMBackendError:
            retry_prompt = repair_prompt or (
                "Return only a valid JSON object with no prose.\n\nOriginal task:\n" + prompt
            )
            repaired = self._request(retry_prompt, expect_json=True)
            return _extract_json_object(repaired)

    def generate_text(self, prompt: str) -> str:
        return self._request(prompt, expect_json=False)


@dataclass
class GeminiBackend:
    provider: str = "gemini"
    mode: str = "provider"
    model: str = "gemini-2.5-pro"
    api_key_env: str = "GEMINI_API_KEY"
    base_url: str = ""
    timeout_sec: float = 60.0
    temperature: float = 0.0
    max_tokens: int = 2000

    def _api_key(self) -> str:
        candidates = [self.api_key_env, "GEMINI_API_KEY", "GOOGLE_API_KEY"]
        for name in candidates:
            value = os.getenv(name, "").strip()
            if value:
                return value
        raise LLMBackendError(
            f"Missing Gemini API key. Checked {', '.join(dict.fromkeys(candidates))}."
        )

    def _model(self):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise LLMBackendError(
                "Gemini backend requires langchain_google_genai to be installed."
            ) from exc
        return ChatGoogleGenerativeAI(
            model=self.model,
            api_key=self._api_key(),
            temperature=self.temperature,
        )

    def generate_json(self, prompt: str, *, repair_prompt: str | None = None) -> Dict[str, Any]:
        model = self._model()
        text = model.invoke(prompt).content
        if not isinstance(text, str):
            text = str(text)
        try:
            return _extract_json_object(text)
        except LLMBackendError:
            retry_prompt = repair_prompt or (
                "Return only a valid JSON object with no prose.\n\nOriginal task:\n" + prompt
            )
            repaired = model.invoke(retry_prompt).content
            if not isinstance(repaired, str):
                repaired = str(repaired)
            return _extract_json_object(repaired)

    def generate_text(self, prompt: str) -> str:
        text = self._model().invoke(prompt).content
        return text if isinstance(text, str) else str(text)


def get_llm(config: Dict[str, Any] | None = None):
    llm_cfg = dict((config or {}).get("llm", {}))
    provider = str(llm_cfg.get("provider", "offline")).strip().lower()
    mode = str(llm_cfg.get("mode", "offline_stub")).strip().lower()
    if mode == "offline_stub":
        return OfflineBackend(provider=provider, mode=mode, model=str(llm_cfg.get("model", "offline-stub")))
    if provider == "openrouter":
        return OpenRouterBackend(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "z-ai/glm-4.5-air:free")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 60.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", llm_cfg.get("max_tokens_structured", 2000))),
            max_retries=int(llm_cfg.get("max_retries", 3)),
        )
    if provider == "bigmodel":
        return BigModelBackend(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "glm-4.7-flashx")),
            api_key_env=str(llm_cfg.get("api_key_env", "BIGMODEL_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://open.bigmodel.cn/api/paas/v4")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 60.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", llm_cfg.get("max_tokens_structured", 2000))),
            max_retries=int(llm_cfg.get("max_retries", 3)),
            thinking_type=str(llm_cfg.get("thinking_type", "disabled")),
            clear_thinking=llm_cfg.get("clear_thinking"),
        )
    if provider == "gemini":
        return GeminiBackend(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "gemini-2.5-pro")),
            api_key_env=str(llm_cfg.get("api_key_env", "GEMINI_API_KEY")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 60.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=int(llm_cfg.get("max_tokens", llm_cfg.get("max_tokens_structured", 2000))),
        )
    raise LLMBackendError(f"Unsupported llm provider={provider}, mode={mode}")
