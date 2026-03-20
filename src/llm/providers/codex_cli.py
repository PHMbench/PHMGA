"""Codex CLI-backed provider implementation for PHM Formal Main runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan

from ..base import LLMProviderError, LLMSchemaError
from ..structured import (
    _codex_reflection_schema,
    _deterministic_param_resolution,
    _json_text_preview,
    _local_param_resolution,
    _param_json_schema,
    _parse_json_object,
    _parse_plan_text_payload,
    _parse_reflection_text_payload,
    _repair_prompt,
)
from ..tracing import append_planner_trace_event, write_planner_text_artifact
from .http_provider import OpenRouterLLM


@dataclass
class CodexCliLLM(OpenRouterLLM):
    """Codex CLI-backed strict-structured client for Formal Main runs."""

    provider: str = "codex_cli"
    mode: str = "provider"
    model: str = "gpt-5.3-codex"
    api_key_env: str = ""
    base_url: str = ""
    app_title: Optional[str] = None
    http_referer: Optional[str] = None
    codex_bin: str = "codex"
    working_dir: str = field(default_factory=lambda: str(Path(__file__).resolve().parents[3]))
    sandbox_mode: str = "read-only"
    reasoning_effort: str = "low"
    planner_smoke_timeout_sec: float = 15.0

    def _resolve_codex_bin(self) -> str:
        binary = shutil.which(self.codex_bin)
        if not binary:
            raise LLMProviderError(
                f"Missing Codex CLI binary '{self.codex_bin}'. Install Codex CLI or add it to PATH."
            )
        return binary

    def _codex_prompt(self, prompt: str, *, structured: bool) -> str:
        prefix = "Respond directly. Do not run tools or shell commands.\n"
        if structured:
            prefix += "Return only the final structured answer that matches the provided schema.\n\n"
        else:
            prefix += "Return only the final markdown answer.\n\n"
        return prefix + prompt

    def _record_transport_event(
        self,
        trace_context: Optional[Dict[str, Any]],
        *,
        stage: str,
        status: str,
        timeout_sec: float,
        elapsed_sec: float,
        stdout_preview: str = "",
        stderr_preview: str = "",
        output_preview: str = "",
        message: Optional[str] = None,
    ) -> None:
        append_planner_trace_event(
            trace_context,
            filename="planner_transport_trace.json",
            provider=self.provider,
            model=self.model,
            event={
                "stage": stage,
                "status": status,
                "timeout_sec": timeout_sec,
                "elapsed_sec": round(elapsed_sec, 3),
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "output_preview": output_preview,
                "message": message,
            },
        )

    def _record_normalization_event(
        self,
        trace_context: Optional[Dict[str, Any]],
        *,
        attempt: str,
        status: str,
        raw_response_file: Optional[str] = None,
        parsed_step_count: Optional[int] = None,
        message: Optional[str] = None,
    ) -> None:
        append_planner_trace_event(
            trace_context,
            filename="planner_normalization_trace.json",
            provider=self.provider,
            model=self.model,
            event={
                "attempt": attempt,
                "status": status,
                "raw_response_file": raw_response_file,
                "parsed_step_count": parsed_step_count,
                "message": message,
            },
        )

    def _codex_exec(
        self,
        *,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        structured_output: bool = False,
        timeout_sec: Optional[float] = None,
        trace_context: Optional[Dict[str, Any]] = None,
        trace_stage: str = "planner",
    ) -> str:
        codex_bin = self._resolve_codex_bin()
        output_file = tempfile.NamedTemporaryFile(prefix="phmga_codex_output_", suffix=".txt", delete=False)
        output_file.close()
        schema_path: Optional[str] = None
        effective_timeout = float(timeout_sec or self.timeout_sec)
        started = time.monotonic()
        try:
            cmd = [
                codex_bin,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-c",
                f'plan_mode_reasoning_effort="{self.reasoning_effort}"',
                "exec",
                "--skip-git-repo-check",
                "--cd",
                self.working_dir,
                "-m",
                self.model,
                "--sandbox",
                self.sandbox_mode,
                "--color",
                "never",
                "-o",
                output_file.name,
            ]
            if schema is not None:
                schema_file = tempfile.NamedTemporaryFile(prefix="phmga_codex_schema_", suffix=".json", delete=False)
                with open(schema_file.name, "w", encoding="utf-8") as handle:
                    json.dump(schema, handle, ensure_ascii=False)
                schema_path = schema_file.name
                cmd.extend(["--output-schema", schema_path])
            cmd.append("-")
            result = subprocess.run(
                cmd,
                input=self._codex_prompt(prompt, structured=(schema is not None or structured_output)),
                text=True,
                capture_output=True,
                timeout=effective_timeout,
                cwd=self.working_dir,
            )
            elapsed = time.monotonic() - started
            stdout_preview = _json_text_preview(result.stdout or "")
            stderr_preview = _json_text_preview(result.stderr or "")
            if result.returncode != 0:
                self._record_transport_event(
                    trace_context,
                    stage=trace_stage,
                    status="returncode_error",
                    timeout_sec=effective_timeout,
                    elapsed_sec=elapsed,
                    stdout_preview=stdout_preview,
                    stderr_preview=stderr_preview,
                    message=f"returncode={result.returncode}",
                )
                raise LLMProviderError(
                    f"{self.provider} exec failed with code {result.returncode}. stderr={stderr_preview!r}"
                )
            output_text = ""
            output_path = Path(output_file.name)
            if output_path.exists():
                output_text = output_path.read_text(encoding="utf-8").strip()
            if not output_text:
                output_text = (result.stdout or "").strip()
            if not output_text:
                self._record_transport_event(
                    trace_context,
                    stage=trace_stage,
                    status="empty_output",
                    timeout_sec=effective_timeout,
                    elapsed_sec=elapsed,
                    stdout_preview=stdout_preview,
                    stderr_preview=stderr_preview,
                    message="provider returned empty output",
                )
                raise LLMSchemaError(f"{self.provider} returned empty output for model={self.model}.")
            self._record_transport_event(
                trace_context,
                stage=trace_stage,
                status="ok",
                timeout_sec=effective_timeout,
                elapsed_sec=elapsed,
                stdout_preview=stdout_preview,
                stderr_preview=stderr_preview,
                output_preview=_json_text_preview(output_text),
            )
            return output_text
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started
            stdout_preview = _json_text_preview(str(getattr(exc, "stdout", "") or getattr(exc, "output", "") or ""))
            stderr_preview = _json_text_preview(str(getattr(exc, "stderr", "") or ""))
            self._record_transport_event(
                trace_context,
                stage=trace_stage,
                status="timeout",
                timeout_sec=effective_timeout,
                elapsed_sec=elapsed,
                stdout_preview=stdout_preview,
                stderr_preview=stderr_preview,
                message="codex exec did not return within timeout window",
            )
            raise LLMProviderError(
                f"{self.provider} exec timed out after {effective_timeout} seconds for model={self.model}."
            ) from exc
        finally:
            try:
                os.unlink(output_file.name)
            except FileNotFoundError:
                pass
            if schema_path:
                try:
                    os.unlink(schema_path)
                except FileNotFoundError:
                    pass

    def _planner_transport_smoke(self, trace_context: Optional[Dict[str, Any]]) -> None:
        smoke_schema = {
            "type": "object",
            "properties": {"ok": {"type": "string"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        smoke_text = self._codex_exec(
            prompt="Return a JSON object with a single field `ok` set to `yes`.",
            schema=smoke_schema,
            timeout_sec=min(self.timeout_sec, self.planner_smoke_timeout_sec),
            trace_context=trace_context,
            trace_stage="planner_smoke",
        )
        parsed = _parse_json_object(
            smoke_text,
            model=self.model,
            structured_mode="json_mode",
        )
        if str(parsed.get("ok", "")).strip().lower() != "yes":
            raise LLMSchemaError(
                f"{self.provider} planner smoke returned an unexpected payload for model={self.model}: {parsed}"
            )

    def generate_step_plan(
        self,
        *,
        prompt: str,
        instruction: str,
        signal_context: SignalContext,
        dag_json: Optional[Dict[str, Any]],
        reflection: Iterable[str],
        operator_catalog_summary: Iterable[Dict[str, Any]],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> StepPlan:
        del instruction, signal_context, dag_json, reflection
        if str((trace_context or {}).get("graph_path", "")).strip().lower() != "ml":
            trace_context = None
        if trace_context is not None:
            self._planner_transport_smoke(trace_context)
        text = self._codex_exec(
            prompt=prompt,
            structured_output=True,
            trace_context=trace_context,
            trace_stage="planner_full",
        )
        normalized_from_attempt = "initial"
        raw_response_file = write_planner_text_artifact(
            trace_context,
            filename="planner_raw_response.txt",
            content=text,
        )
        try:
            payload = _parse_plan_text_payload(
                text,
                model=self.model,
                provider=self.provider,
                allow_text_fallback=False,
            )
        except LLMSchemaError as first_error:
            self._record_normalization_event(
                trace_context,
                attempt="initial",
                status="schema_error",
                raw_response_file=raw_response_file,
                message=str(first_error),
            )
            if not self.retry_once:
                raise
            repair_text = self._codex_exec(
                prompt=_repair_prompt(task="plan", original_prompt=prompt, raw_response=text),
                structured_output=True,
                trace_context=trace_context,
                trace_stage="planner_repair",
            )
            repair_response_file = write_planner_text_artifact(
                trace_context,
                filename="planner_repair_response.txt",
                content=repair_text,
            )
            try:
                payload = _parse_plan_text_payload(
                    repair_text,
                    model=self.model,
                    provider=self.provider,
                    allow_text_fallback=True,
                )
                normalized_from_attempt = "repair"
                raw_response_file = repair_response_file
                text = repair_text
            except LLMSchemaError as second_error:
                self._record_normalization_event(
                    trace_context,
                    attempt="repair",
                    status="schema_error",
                    raw_response_file=repair_response_file,
                    message=f"{first_error} | repair_attempt_failed={second_error}",
                )
                raise LLMSchemaError(f"{first_error} | repair_attempt_failed={second_error}") from second_error
        self._record_normalization_event(
            trace_context,
            attempt=normalized_from_attempt,
            status="normalized",
            raw_response_file=raw_response_file,
            parsed_step_count=len(payload.get("plan", [])),
            message=f"text_preview={_json_text_preview(text)!r}",
        )
        return StepPlan.model_validate(payload)

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

        schema = {
            "type": "object",
            "properties": {name: _param_json_schema(param_schema.get(name, "")) for name in missing_tunable},
            "required": missing_tunable,
            "additionalProperties": False,
        }
        provider_params = _parse_json_object(
            self._codex_exec(prompt=prompt, schema=schema),
            model=self.model,
            structured_mode="json_mode",
        )
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
        text = self._codex_exec(prompt=prompt, schema=_codex_reflection_schema())
        try:
            payload = _parse_reflection_text_payload(text, model=self.model, provider=self.provider)
        except LLMSchemaError as first_error:
            if not self.retry_once:
                raise
            repair_text = self._codex_exec(
                prompt=_repair_prompt(task="reflect", original_prompt=prompt, raw_response=text),
                schema=_codex_reflection_schema(),
            )
            try:
                payload = _parse_reflection_text_payload(repair_text, model=self.model, provider=self.provider)
            except LLMSchemaError as second_error:
                raise LLMSchemaError(f"{first_error} | repair_attempt_failed={second_error}") from second_error
        return ReflectionResult.model_validate(payload)

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
        return self._codex_exec(prompt=prompt).strip() + "\n"


__all__ = ["CodexCliLLM"]
