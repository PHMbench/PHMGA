"""HTTP-backed LLM providers for structured PHM agents."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

import httpx

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan

from ..base import LLMProviderError, LLMSchemaError
from ..structured import (
    _deterministic_param_resolution,
    _extract_message_text,
    _local_param_resolution,
    _parse_json_object,
    _parse_plan_text_payload,
    _parse_reflection_text_payload,
    _repair_prompt,
    _structured_mode_for_model,
)


@dataclass
class OpenRouterLLM:
    """Chat-completions provider client for structured agents plus report rendering."""

    provider: str = "openrouter"
    mode: str = "provider"
    model: str = "openai/gpt-4.1-mini"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_sec: float = 30.0
    temperature: float = 0.0
    max_tokens_structured: int = 800
    max_tokens_report: int = 2000
    retry_once: bool = True
    http_referer: Optional[str] = None
    app_title: Optional[str] = "PHMGA"
    http_client: Optional[httpx.Client] = field(default=None, repr=False)

    def _api_key(self) -> str:
        api_key = os.getenv(self.api_key_env, "").strip()
        if not api_key:
            raise LLMProviderError(
                f"Missing API key for provider '{self.provider}'. Expected env var {self.api_key_env}."
            )
        return api_key

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def _client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        self.http_client = httpx.Client(timeout=self.timeout_sec)
        return self.http_client

    def _request_text(self, *, prompt: str, max_tokens: int, expect_json: bool) -> str:
        structured_mode = _structured_mode_for_model(self.model)
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if expect_json and structured_mode == "json_mode":
            body["response_format"] = {"type": "json_object"}
        attempts = 2 if self.retry_once else 1
        response: Optional[httpx.Response] = None
        last_error: Optional[Exception] = None
        for attempt_index in range(attempts):
            try:
                response = self._client().post(
                    f"{self.base_url.rstrip('/')}/chat/completions",
                    headers=self._headers(),
                    json=body,
                )
                response.raise_for_status()
                last_error = None
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                retryable = status_code == 429 or 500 <= status_code < 600
                last_error = exc
                if retryable and attempt_index + 1 < attempts:
                    time.sleep(0.05)
                    continue
                raise LLMProviderError(
                    f"{self.provider} request failed with status {status_code}: {exc.response.text}"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt_index + 1 < attempts:
                    time.sleep(0.05)
                    continue
                raise LLMProviderError(f"{self.provider} transport error: {exc}") from exc

        if response is None:
            raise LLMProviderError(f"{self.provider} request failed before a response was received: {last_error}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMSchemaError(f"Provider returned non-JSON HTTP payload: {exc}") from exc

        return _extract_message_text(payload)

    def _request_json_object(self, *, prompt: str, max_tokens: int) -> Dict[str, Any]:
        structured_mode = _structured_mode_for_model(self.model)
        effective_prompt = prompt
        if structured_mode == "text_mode":
            effective_prompt = (
                "You are a JSON API. Output ONLY valid JSON. No text before or after.\n\n"
                "Example format:\n"
                '{\n  "field1": "value1",\n  "field2": "value2"\n}\n\n'
                f"Task:\n{prompt}\n\n"
                "Response (JSON only, start with {{):"
            )
        text = self._request_text(
            prompt=effective_prompt,
            max_tokens=max_tokens,
            expect_json=(structured_mode == "json_mode"),
        )
        return _parse_json_object(text, model=self.model, structured_mode=structured_mode)

    def generate_step_plan(
        self,
        *,
        prompt: str,
        instruction: str,
        signal_context: SignalContext,
        dag_json: Optional[Dict[str, Any]],
        reflection: Iterable[str],
        operator_catalog_summary: Iterable[Dict[str, Any]],
    ) -> StepPlan:
        del instruction, signal_context, dag_json, reflection, operator_catalog_summary
        text = self._request_text(prompt=prompt, max_tokens=self.max_tokens_structured, expect_json=True)
        try:
            payload = _parse_plan_text_payload(text, model=self.model, provider=self.provider)
        except LLMSchemaError as first_error:
            if not self.retry_once:
                raise
            repair_text = self._request_text(
                prompt=_repair_prompt(task="plan", original_prompt=prompt, raw_response=text),
                max_tokens=self.max_tokens_structured,
                expect_json=True,
            )
            try:
                payload = _parse_plan_text_payload(repair_text, model=self.model, provider=self.provider)
            except LLMSchemaError as second_error:
                raise LLMSchemaError(f"{first_error} | repair_attempt_failed={second_error}") from second_error
        try:
            return StepPlan.model_validate(payload)
        except Exception as exc:
            raise LLMSchemaError(f"Planner response failed StepPlan validation: {exc}") from exc

    def resolve_missing_params(
        self,
        *,
        prompt: str,
        op_name: str,
        param_schema: Dict[str, str],
        param_defaults: Dict[str, Any],
        param_docs: Dict[str, str],
        llm_tunable_params: List[str],
        provided_params: Dict[str, Any],
        signal_context: SignalContext,
        parent_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        del param_docs
        resolved = _local_param_resolution(
            param_schema=param_schema,
            param_defaults=param_defaults,
            provided_params=provided_params,
            signal_context=signal_context,
            parent_summaries=parent_summaries,
        )
        missing = [param_name for param_name in param_schema if param_name not in resolved]
        missing_non_tunable = [param_name for param_name in missing if param_name not in llm_tunable_params]
        if missing_non_tunable:
            raise ValueError(f"Missing required parameter(s) {missing_non_tunable} for op '{op_name}'.")
        missing_tunable = [param_name for param_name in missing if param_name in llm_tunable_params]
        if not missing_tunable:
            return _deterministic_param_resolution(
                op_name=op_name,
                param_schema=param_schema,
                param_defaults=param_defaults,
                llm_tunable_params=llm_tunable_params,
                provided_params=provided_params,
                signal_context=signal_context,
                parent_summaries=parent_summaries,
            )
        provider_params = self._request_json_object(prompt=prompt, max_tokens=self.max_tokens_structured)
        unexpected = sorted(set(provider_params) - set(missing_tunable))
        if unexpected:
            raise LLMSchemaError(
                f"Provider attempted to return non-requested params for '{op_name}': {unexpected}"
            )
        return _deterministic_param_resolution(
            op_name=op_name,
            param_schema=param_schema,
            param_defaults=param_defaults,
            llm_tunable_params=llm_tunable_params,
            provided_params=provided_params,
            signal_context=signal_context,
            parent_summaries=parent_summaries,
            provider_params=provider_params,
        )

    def reflect_workflow(
        self,
        *,
        prompt: str,
        instruction: str,
        stage: str,
        dag_blueprint: Dict[str, Any],
        dag_quality_summary: Dict[str, Any],
        issues_summary: str,
        min_depth: int,
        min_width: int,
        max_depth: int,
        current_depth: int,
        execution_gaps: List[ExecutionGap],
    ) -> ReflectionResult:
        del instruction, stage, dag_blueprint, dag_quality_summary, issues_summary
        del min_depth, min_width, max_depth, current_depth, execution_gaps
        text = self._request_text(prompt=prompt, max_tokens=self.max_tokens_structured, expect_json=True)
        try:
            payload = _parse_reflection_text_payload(text, model=self.model, provider=self.provider)
        except LLMSchemaError as first_error:
            if not self.retry_once:
                raise
            repair_text = self._request_text(
                prompt=_repair_prompt(task="reflect", original_prompt=prompt, raw_response=text),
                max_tokens=self.max_tokens_structured,
                expect_json=True,
            )
            try:
                payload = _parse_reflection_text_payload(repair_text, model=self.model, provider=self.provider)
            except LLMSchemaError as second_error:
                raise LLMSchemaError(f"{first_error} | repair_attempt_failed={second_error}") from second_error
        try:
            return ReflectionResult.model_validate(payload)
        except Exception as exc:
            raise LLMSchemaError(f"Reflector response failed ReflectionResult validation: {exc}") from exc

    def render_report(
        self,
        *,
        prompt: str,
        instruction: str,
        dataset_name: str,
        graph_path: str,
        compiled_manifest: Dict[str, Any],
        path_artifacts: Dict[str, Any],
        reflection_summary: Dict[str, Any],
        dag_quality_summary: Dict[str, Any],
        review_context: Dict[str, Any],
        step_plan: Dict[str, Any],
    ) -> str:
        del instruction, dataset_name, graph_path, compiled_manifest, path_artifacts
        del reflection_summary, dag_quality_summary, review_context, step_plan
        return self._request_text(prompt=prompt, max_tokens=self.max_tokens_report, expect_json=False).strip() + "\n"


@dataclass
class OpenAICodexLLM(OpenRouterLLM):
    """OpenAI-backed strict-structured client, intended for planning and reflection."""

    provider: str = "openai"
    mode: str = "provider"
    model: str = "gpt-5.3-codex"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    app_title: Optional[str] = None


__all__ = ["OpenAICodexLLM", "OpenRouterLLM"]
