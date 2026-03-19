"""Codex CLI-backed provider implementation for PHM Formal Main runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan

from ..base import LLMProviderError, LLMSchemaError
from ..structured import (
    _codex_reflection_schema,
    _codex_step_plan_schema,
    _deterministic_param_resolution,
    _json_text_preview,
    _local_param_resolution,
    _param_json_schema,
    _parse_json_object,
    _parse_plan_text_payload,
    _parse_reflection_text_payload,
    _repair_prompt,
)
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

    def _codex_exec(self, *, prompt: str, schema: Optional[Dict[str, Any]] = None) -> str:
        codex_bin = self._resolve_codex_bin()
        output_file = tempfile.NamedTemporaryFile(prefix="phmga_codex_output_", suffix=".txt", delete=False)
        output_file.close()
        schema_path: Optional[str] = None
        try:
            cmd = [
                codex_bin,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-c",
                f'plan_mode_reasoning_effort="{self.reasoning_effort}"',
                "exec",
                "--ephemeral",
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
                input=self._codex_prompt(prompt, structured=schema is not None),
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
                cwd=self.working_dir,
            )
            if result.returncode != 0:
                stderr_preview = _json_text_preview(result.stderr or result.stdout or "")
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
                raise LLMSchemaError(f"{self.provider} returned empty output for model={self.model}.")
            return output_text
        except subprocess.TimeoutExpired as exc:
            raise LLMProviderError(
                f"{self.provider} exec timed out after {self.timeout_sec} seconds for model={self.model}."
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
        text = self._codex_exec(prompt=prompt, schema=_codex_step_plan_schema())
        try:
            payload = _parse_plan_text_payload(text, model=self.model, provider=self.provider)
        except LLMSchemaError as first_error:
            if not self.retry_once:
                raise
            repair_text = self._codex_exec(
                prompt=_repair_prompt(task="plan", original_prompt=prompt, raw_response=text),
                schema=_codex_step_plan_schema(),
            )
            try:
                payload = _parse_plan_text_payload(repair_text, model=self.model, provider=self.provider)
            except LLMSchemaError as second_error:
                raise LLMSchemaError(f"{first_error} | repair_attempt_failed={second_error}") from second_error
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
