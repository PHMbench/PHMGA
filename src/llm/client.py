"""LLM client implementations for the rebuilt workflow agents.

The repository keeps an offline stub as the default deterministic baseline,
while also exposing a provider-backed OpenRouter path for planner, parameter
completion, reflection, and reporting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

import httpx

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan


def _leaf_node_ids(dag_json: Optional[Dict[str, Any]]) -> List[str]:
    if not dag_json:
        return []
    nodes = {node["node_id"] for node in dag_json.get("nodes", [])}
    parents = {edge["source"] for edge in dag_json.get("edges", [])}
    leaves = sorted(nodes - parents)
    return leaves


def _catalog_entry(operator_catalog_summary: Iterable[Dict[str, Any]], op_name: str) -> Optional[Dict[str, Any]]:
    normalized = op_name.strip().lower()
    for item in operator_catalog_summary:
        if str(item.get("op_name", "")).strip().lower() == normalized:
            return item
    return None


def _supports(operator_catalog_summary: Iterable[Dict[str, Any]], op_name: str) -> bool:
    return _catalog_entry(operator_catalog_summary, op_name) is not None


def _planned_node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{parent.replace(',', '__')}"


def _append_step(steps: list[dict[str, Any]], parent: str, op_name: str, params: dict[str, Any]) -> str:
    steps.append({"parent": parent, "op_name": op_name, "params": params})
    return _planned_node_id(len(steps), op_name, parent)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMSchemaError("Provider response did not include any choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
        if chunks:
            return "".join(chunks)
    raise LLMSchemaError("Provider response did not include textual message content.")


def _parse_json_object(text: str) -> Dict[str, Any]:
    normalized = _strip_json_fence(text)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised via tests
        raise LLMSchemaError(f"Provider response was not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMSchemaError("Provider response must decode to a JSON object.")
    return parsed


def _derived_param_candidates(signal_context: SignalContext, parent_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    window_length = int(signal_context.window_shape[-1]) if signal_context.window_shape else 128
    patch_length = max(16, min(128, max(window_length // 8, 16)))
    derived: Dict[str, Any] = {
        "fs": signal_context.sampling_rate,
        "sampling_rate": signal_context.sampling_rate,
        "nperseg": max(16, min(128, window_length // 4 or window_length)),
        "noverlap": max(8, min(64, window_length // 8 or 8)),
        "patch_length": patch_length,
        "stride": max(8, patch_length // 2),
        "band_low_hz": max(5.0, signal_context.sampling_rate * 0.02),
        "band_high_hz": max(20.0, signal_context.sampling_rate * 0.2),
        "low_cut_hz": max(5.0, signal_context.sampling_rate * 0.02),
        "high_cut_hz": max(20.0, signal_context.sampling_rate * 0.2),
        "threshold": 0.5,
        "mode": "bandpass",
        "order": 4,
        "axis": 0,
    }
    if parent_summaries and any("shape" in item for item in parent_summaries):
        derived["parent_count"] = len(parent_summaries)
    return derived


def _deterministic_param_resolution(
    *,
    op_name: str,
    param_schema: Dict[str, str],
    param_defaults: Dict[str, Any],
    llm_tunable_params: List[str],
    provided_params: Dict[str, Any],
    signal_context: SignalContext,
    parent_summaries: List[Dict[str, Any]],
    provider_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(provided_params)
    derived = _derived_param_candidates(signal_context, parent_summaries)

    for param_name in param_schema:
        if param_name in resolved:
            continue
        if param_name in derived:
            resolved[param_name] = derived[param_name]
            continue
        if param_name in param_defaults:
            resolved[param_name] = param_defaults[param_name]
            continue

    if provider_params:
        illegal = sorted(set(provider_params) - set(llm_tunable_params))
        if illegal:
            raise LLMSchemaError(
                f"Provider attempted to set non-tunable params for '{op_name}': {illegal}"
            )
        for param_name, value in provider_params.items():
            if param_name not in resolved:
                resolved[param_name] = value

    missing = [param_name for param_name in param_schema if param_name not in resolved]
    if missing:
        raise ValueError(f"Missing required parameter(s) {missing} for op '{op_name}'.")
    return resolved


def _local_param_resolution(
    *,
    param_schema: Dict[str, str],
    param_defaults: Dict[str, Any],
    provided_params: Dict[str, Any],
    signal_context: SignalContext,
    parent_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved = dict(provided_params)
    derived = _derived_param_candidates(signal_context, parent_summaries)
    for param_name in param_schema:
        if param_name in resolved:
            continue
        if param_name in derived:
            resolved[param_name] = derived[param_name]
            continue
        if param_name in param_defaults:
            resolved[param_name] = param_defaults[param_name]
    return resolved


@runtime_checkable
class LLMClient(Protocol):
    """Common interface shared by offline and provider-backed clients."""

    provider: str
    mode: str
    model: str
    api_key_env: str

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
        ...

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
        ...

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
        ...

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
        ...


class LLMProviderError(RuntimeError):
    """Raised when the provider transport fails or is misconfigured."""


class LLMSchemaError(ValueError):
    """Raised when a provider response cannot satisfy the expected schema."""


@dataclass
class OfflineLLM:
    """Deterministic stand-in for the provider-backed client."""

    provider: str = "openrouter"
    mode: str = "offline_stub"
    model: str = "offline-stub"
    api_key_env: str = "OPENROUTER_API_KEY"

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
        """Return a deterministic NVTA-style plan for the current DAG state."""

        del prompt, instruction, reflection
        steps: list[dict[str, Any]] = []
        if not dag_json or not dag_json.get("nodes"):
            roots = signal_context.root_node_ids
            normalized: dict[str, str] = {}
            for root in roots:
                if _supports(operator_catalog_summary, "normalize"):
                    normalized[root] = _append_step(steps, root, "normalize", {"eps": 1e-6})
                else:
                    normalized[root] = root

            feature_parents: list[str] = []
            primary_root = roots[0]
            primary_signal = normalized[primary_root]

            if _supports(operator_catalog_summary, "filter"):
                filtered = _append_step(steps, primary_signal, "filter", {})
            else:
                filtered = primary_signal
            if _supports(operator_catalog_summary, "hilbert_envelope"):
                envelope = _append_step(steps, filtered, "hilbert_envelope", {})
            else:
                envelope = filtered
            if _supports(operator_catalog_summary, "kurtosis"):
                feature_parents.append(_append_step(steps, envelope, "kurtosis", {}))

            if _supports(operator_catalog_summary, "stft"):
                stft_node = _append_step(steps, primary_signal, "stft", {})
                if _supports(operator_catalog_summary, "spectral_centroid"):
                    feature_parents.append(_append_step(steps, stft_node, "spectral_centroid", {}))

            if _supports(operator_catalog_summary, "patch"):
                patch_node = _append_step(steps, primary_signal, "patch", {})
                if _supports(operator_catalog_summary, "kurtosis"):
                    feature_parents.append(_append_step(steps, patch_node, "kurtosis", {}))

            for root in roots[:2]:
                source = normalized[root]
                if _supports(operator_catalog_summary, "psd"):
                    psd_node = _append_step(steps, source, "psd", {})
                    if _supports(operator_catalog_summary, "band_power"):
                        feature_parents.append(_append_step(steps, psd_node, "band_power", {}))
                elif _supports(operator_catalog_summary, "fft"):
                    fft_node = _append_step(steps, source, "fft", {})
                    if _supports(operator_catalog_summary, "rms"):
                        feature_parents.append(_append_step(steps, fft_node, "rms", {}))

            if len(roots) > 1 and _supports(operator_catalog_summary, "cross_correlation"):
                cross_parent = f"{normalized[roots[0]]},{normalized[roots[1]]}"
                feature_parents.append(_append_step(steps, cross_parent, "cross_correlation", {}))
            elif len(roots) > 1 and _supports(operator_catalog_summary, "crest_factor"):
                feature_parents.append(_append_step(steps, normalized[roots[1]], "crest_factor", {}))

            fused_node: str | None = None
            if len(feature_parents) >= 2 and _supports(operator_catalog_summary, "concatenate"):
                fused_node = _append_step(steps, ",".join(feature_parents[:4]), "concatenate", {"axis": 0})

            if _supports(operator_catalog_summary, "threshold"):
                decision_parent = fused_node or (feature_parents[0] if feature_parents else primary_signal)
                _append_step(steps, decision_parent, "threshold", {})
            return StepPlan.model_validate({"plan": steps})

        leaves = _leaf_node_ids(dag_json) or signal_context.root_node_ids
        for leaf in leaves:
            if leaf.startswith("stft_") and _supports(operator_catalog_summary, "spectral_centroid"):
                _append_step(steps, leaf, "spectral_centroid", {})
            elif leaf.startswith("patch_") and _supports(operator_catalog_summary, "kurtosis"):
                _append_step(steps, leaf, "kurtosis", {})
            elif leaf.startswith("psd_") and _supports(operator_catalog_summary, "band_power"):
                _append_step(steps, leaf, "band_power", {})
            elif leaf.startswith("normalize_") and _supports(operator_catalog_summary, "filter"):
                _append_step(steps, leaf, "filter", {})
            elif _supports(operator_catalog_summary, "threshold") and not leaf.startswith("threshold_"):
                _append_step(steps, leaf, "threshold", {})
            elif _supports(operator_catalog_summary, "normalize"):
                _append_step(steps, leaf, "normalize", {"eps": 1e-6})
        return StepPlan.model_validate({"plan": steps})

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
        """Resolve operator params deterministically from explicit/local context."""

        del prompt, param_docs
        return _deterministic_param_resolution(
            op_name=op_name,
            param_schema=param_schema,
            param_defaults=param_defaults,
            llm_tunable_params=llm_tunable_params,
            provided_params=provided_params,
            signal_context=signal_context,
            parent_summaries=parent_summaries,
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
        """Return a structured NVTA-style review result."""

        del prompt, instruction, stage, min_width
        missing_operators = sorted(
            {
                gap.op_name
                for gap in execution_gaps
                if "Unknown" in gap.message or "unsupported" in gap.message.lower()
            }
        )
        shape_risks = [gap.message for gap in execution_gaps if "shape" in gap.message.lower()]
        structural_warnings: list[str] = []
        if not dag_blueprint.get("nodes"):
            return ReflectionResult(
                decision="halt",
                reason="DAG blueprint is empty.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=["No nodes were materialized."],
            )
        if execution_gaps:
            structural_warnings.extend(gap.message for gap in execution_gaps)
            fatal = any(not gap.recoverable for gap in execution_gaps)
            return ReflectionResult(
                decision="need_replan" if fatal else "need_patch",
                reason=issues_summary or "Execution gaps prevent the current plan from completing cleanly.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        quality_issues = list(dag_quality_summary.get("issues", []))
        structural_warnings.extend(quality_issues)
        recommendation_hint = str(dag_quality_summary.get("recommendation_hint", ""))
        if recommendation_hint == "halt_candidate":
            return ReflectionResult(
                decision="halt",
                reason="The DAG quality summary marked the current round as halt_candidate.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        if recommendation_hint == "replan_candidate" or current_depth > max_depth:
            return ReflectionResult(
                decision="need_replan",
                reason="The DAG quality summary indicates that the current round should be replanned.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        depth_ok = bool(dag_quality_summary.get("depth_ok", current_depth >= min_depth))
        if recommendation_hint == "patch_candidate" or not depth_ok:
            return ReflectionResult(
                decision="need_patch",
                reason="The workflow is structurally healthy but the DAG quality summary recommends another patch round.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings or ["Continue expanding the DAG."],
            )
        return ReflectionResult(
            decision="finish",
            reason="The DAG is structurally valid and satisfies the current planning target.",
            missing_operators=missing_operators,
            shape_risks=shape_risks,
            structural_warnings=structural_warnings,
        )

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
        """Generate a deterministic markdown report with path-specific sections."""

        del prompt

        def _keys_or_len(value: Any) -> str:
            if isinstance(value, dict):
                return ", ".join(sorted(value.keys()))
            if isinstance(value, list):
                return f"{len(value)} entries"
            return "n/a"

        lines = [
            f"# PHMGA Report: {dataset_name} / {graph_path}",
            "",
            "## Summary",
            f"- Instruction: {instruction}",
            f"- DAG hash: `{compiled_manifest['dag_hash']}`",
            f"- Path: {graph_path}",
            f"- Reflection decision: {reflection_summary.get('decision', 'unknown')}",
            "",
            "## Workflow Plan",
            f"- Planned steps: {len(step_plan.get('plan', []))}",
            f"- Workflow rounds: {review_context.get('round_count', 'n/a')}",
        ]
        lines.extend(
            [
                "",
                "## DAG Quality",
                f"- Depth: {dag_quality_summary.get('current_depth', 'n/a')} / min={dag_quality_summary.get('min_depth', 'n/a')} / max={dag_quality_summary.get('max_depth', 'n/a')}",
                f"- Feature nodes: {dag_quality_summary.get('feature_node_count', 'n/a')}",
                f"- Multi nodes: {dag_quality_summary.get('multi_node_count', 'n/a')}",
                f"- Execution gaps: {dag_quality_summary.get('execution_gap_count', 'n/a')}",
                f"- NaN ratio: {dag_quality_summary.get('nan_ratio', 'n/a')}",
                f"- Zero-variance ratio: {dag_quality_summary.get('zero_variance_ratio', 'n/a')}",
                f"- Proxy probe enabled: {dag_quality_summary.get('proxy_probe_enabled', False)}",
                f"- Proxy probe macro_f1: {dag_quality_summary.get('proxy_probe_macro_f1', 'n/a')}",
            ]
        )
        decision_outputs = path_artifacts.get("decision_side_outputs", {})
        if isinstance(decision_outputs, dict) and decision_outputs:
            lines.extend(
                [
                    "",
                    "## Decision Side Outputs",
                    f"- Decision nodes: {len(decision_outputs)}",
                    f"- Keys: {', '.join(sorted(decision_outputs.keys()))}",
                ]
            )
        if graph_path == "dag_only":
            lines.extend(
                [
                    "",
                    "## DAG Evidence",
                    f"- Node inventory: {len(path_artifacts.get('node_inventory', []))}",
                    f"- Edge inventory: {len(path_artifacts.get('edge_inventory', []))}",
                    f"- Method description: {path_artifacts.get('method_description', 'n/a')}",
                ]
            )
        elif graph_path == "ml":
            lines.extend(
                [
                    "",
                    "## ML Evidence",
                    f"- Execution nodes: {len(path_artifacts.get('feature_pipeline', {}).get('execution_nodes', []))}",
                    f"- Output specs: {len(path_artifacts.get('feature_pipeline', {}).get('output_specs', []))}",
                    f"- Output policy: {path_artifacts.get('feature_pipeline', {}).get('output_policy', 'n/a')}",
                    f"- Metrics keys: {', '.join(sorted(path_artifacts.get('metrics', {}).keys()))}",
                    f"- Importance keys: {', '.join(sorted(path_artifacts.get('importance', {}).keys()))}",
                    f"- Similarity artifact keys: {', '.join(sorted(path_artifacts.get('similarity_artifacts', {}).keys()))}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Torch Evidence",
                    f"- Build plan keys: {_keys_or_len(path_artifacts.get('model_build_plan', {}))}",
                    f"- Training curves: {_keys_or_len(path_artifacts.get('training_curves', {}))}",
                    f"- Checkpoint keys: {_keys_or_len(path_artifacts.get('checkpoint', {}))}",
                    f"- Similarity artifact keys: {', '.join(sorted(path_artifacts.get('similarity_artifacts', {}).keys()))}",
                ]
            )
        lines.extend(
            [
                "",
                "## Review Context",
                f"- Stage: {review_context.get('stage', 'FINAL_REPORT')}",
                f"- Current depth: {review_context.get('current_depth', 'n/a')}",
                f"- Issues summary: {review_context.get('issues_summary', '')}",
                f"- Quality issues: {', '.join(dag_quality_summary.get('issues', []))}",
            ]
        )
        return "\n".join(lines) + "\n"


@dataclass
class OpenRouterLLM:
    """OpenRouter-backed client for planner, param completion, reflection, and reporting."""

    provider: str = "openrouter"
    mode: str = "provider"
    model: str = "openai/gpt-4.1-mini"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_sec: float = 30.0
    temperature: float = 0.0
    max_tokens_structured: int = 800
    max_tokens_report: int = 2000
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
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if expect_json:
            body["response_format"] = {"type": "json_object"}

        try:
            response = self._client().post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"OpenRouter request failed with status {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"OpenRouter transport error: {exc}") from exc

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise LLMSchemaError(f"Provider returned non-JSON HTTP payload: {exc}") from exc
        return _extract_message_text(payload)

    def _request_json_object(self, *, prompt: str, max_tokens: int) -> Dict[str, Any]:
        return _parse_json_object(self._request_text(prompt=prompt, max_tokens=max_tokens, expect_json=True))

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
        payload = self._request_json_object(prompt=prompt, max_tokens=self.max_tokens_structured)
        try:
            return StepPlan.model_validate(payload)
        except Exception as exc:  # pragma: no cover - exercised via tests
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
        missing_tunable = [
            param_name
            for param_name in missing
            if param_name in llm_tunable_params
        ]
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
        payload = self._request_json_object(prompt=prompt, max_tokens=self.max_tokens_structured)
        try:
            return ReflectionResult.model_validate(payload)
        except Exception as exc:  # pragma: no cover - exercised via tests
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


def get_llm(config: Dict[str, Any]) -> LLMClient:
    """Resolve the configured LLM client."""

    llm_cfg = dict(config.get("llm", {}))
    provider = str(llm_cfg.get("provider", "openrouter"))
    mode = str(llm_cfg.get("mode", "offline_stub"))
    if mode == "offline_stub":
        return OfflineLLM(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "offline-stub")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
        )
    if mode == "provider" and provider == "openrouter":
        return OpenRouterLLM(
            provider=provider,
            mode=mode,
            model=str(llm_cfg.get("model", "openai/gpt-4.1-mini")),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://openrouter.ai/api/v1")),
            timeout_sec=float(llm_cfg.get("timeout_sec", 30.0)),
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens_structured=int(llm_cfg.get("max_tokens_structured", 800)),
            max_tokens_report=int(llm_cfg.get("max_tokens_report", 2000)),
            http_referer=str(llm_cfg.get("http_referer", "")).strip() or None,
            app_title=str(llm_cfg.get("app_title", "PHMGA")).strip() or None,
        )
    raise ValueError(f"Unsupported llm configuration: provider={provider}, mode={mode}")


__all__ = [
    "LLMClient",
    "LLMProviderError",
    "LLMSchemaError",
    "OfflineLLM",
    "OpenRouterLLM",
    "get_llm",
]
