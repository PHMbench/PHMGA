"""Base contracts and shared error types for PHM LLM clients."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan


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
        trace_context: Optional[Dict[str, Any]] = None,
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


__all__ = ["LLMClient", "LLMProviderError", "LLMSchemaError"]
