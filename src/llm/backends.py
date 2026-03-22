"""Minimal OpenRouter, Gemini, and offline backends."""

from __future__ import annotations

import json
import os
import re
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

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise LLMBackendError(f"Missing API key in env var {self.api_key_env}.")
        return api_key

    def _request(self, prompt: str, *, expect_json: bool) -> str:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if expect_json:
            body["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                response = client.post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                )
                response.raise_for_status()
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
