"""Reflection agent for validating the current workflow state."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.configuration import Configuration
from src.llm import LLMClient, LLMProviderError, LLMSchemaError
from src.model import LangChainLLMAdapter, get_llm
from src.prompts import render_reflect_prompt
from src.states import PHMState, ReflectionResult


def _dag_depth(state: PHMState) -> int:
    if not state.dag or not state.dag.nodes:
        return 0
    depth_by_node: dict[str, int] = {}
    for node in state.dag.nodes:
        if not node.parents:
            depth_by_node[node.node_id] = 1
        else:
            depth_by_node[node.node_id] = 1 + max(depth_by_node[parent] for parent in node.parents)
    return max(depth_by_node.values(), default=0)


def _resolve_llm(state: PHMState, llm: LLMClient | None) -> LangChainLLMAdapter:
    if llm is not None:
        return LangChainLLMAdapter(llm)
    return get_llm(Configuration.from_runtime_config(state.runtime_config))


def _apply_quality_override(result: ReflectionResult, dag_quality_summary: dict) -> ReflectionResult:
    """Do not allow provider judgment to ignore deterministic DAG quality gates."""

    hint = str(dag_quality_summary.get("recommendation_hint", "")).strip()
    if result.decision == "halt" or hint in {"", "finish_candidate"}:
        return result

    forced_decision = result.decision
    if hint == "halt_candidate" and result.decision == "finish":
        forced_decision = "halt"
    elif hint == "replan_candidate" and result.decision in {"finish", "need_patch"}:
        forced_decision = "need_replan"
    elif hint == "patch_candidate" and result.decision == "finish":
        forced_decision = "need_patch"

    if forced_decision == result.decision:
        return result

    issues = list(dag_quality_summary.get("issues", []) or [])
    dataset_level = dag_quality_summary.get("dataset_level", {})
    if isinstance(dataset_level, dict):
        issues.extend(list(dataset_level.get("issues", []) or []))
    issue_preview = "; ".join(str(item) for item in issues[:3]) or f"quality recommendation was {hint}"
    warnings = list(result.structural_warnings)
    warnings.append(f"deterministic_quality_override: {hint}; {issue_preview}")
    return result.model_copy(
        update={
            "decision": forced_decision,
            "reason": f"{forced_decision} required by DAG quality gate: {issue_preview}",
            "structural_warnings": warnings,
        }
    )


def _quality_issue_preview(dag_quality_summary: dict) -> str:
    issues = list(dag_quality_summary.get("issues", []) or [])
    dataset_level = dag_quality_summary.get("dataset_level", {})
    if isinstance(dataset_level, dict):
        issues.extend(list(dataset_level.get("issues", []) or []))
    return "; ".join(str(item) for item in issues[:3]) or "no quality issue details available"


def _fallback_reflection_from_quality(dag_quality_summary: dict, exc: LLMProviderError | LLMSchemaError) -> ReflectionResult:
    hint = str(dag_quality_summary.get("recommendation_hint", "")).strip()
    decision_by_hint = {
        "finish_candidate": "finish",
        "patch_candidate": "need_patch",
        "replan_candidate": "need_replan",
        "halt_candidate": "halt",
    }
    decision = decision_by_hint.get(hint)
    if decision is None:
        raise exc
    issue_preview = _quality_issue_preview(dag_quality_summary)
    return ReflectionResult(
        decision=decision,
        reason=f"{decision} selected by deterministic quality fallback after provider error: {issue_preview}",
        missing_operators=[],
        shape_risks=[],
        structural_warnings=[
            f"provider_reflection_fallback: {type(exc).__name__}; recommendation_hint={hint}; {issue_preview}"
        ],
    )


def reflect_agent(state: PHMState, llm: LLMClient | None = None) -> PHMState:
    """Review the current DAG, execution gaps, and planning progress."""

    dag_blueprint = state.dag.model_dump() if state.dag else {"nodes": [], "edges": []}
    issues_summary = "\n".join(gap.message for gap in state.execution_gaps)
    current_depth = _dag_depth(state)
    min_depth = int(state.data_context.get("min_depth", 2))
    min_width = int(state.data_context.get("min_width", 1))
    max_depth = int(state.data_context.get("max_depth", 8))
    stage = str(state.data_context.get("stage", "POST_EXECUTE"))
    prompt = render_reflect_prompt(
        instruction=state.user_instruction,
        stage=stage,
        dag_blueprint=dag_blueprint,
        dag_quality_summary=state.dag_quality_summary,
        issues_summary=issues_summary,
        min_depth=min_depth,
        min_width=min_width,
        max_depth=max_depth,
        current_depth=current_depth,
    )
    llm_adapter = _resolve_llm(state, llm)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm_adapter.bind_task(
        "reflect",
        instruction=state.user_instruction,
        stage=stage,
        dag_blueprint=dag_blueprint,
        dag_quality_summary=state.dag_quality_summary,
        issues_summary=issues_summary,
        min_depth=min_depth,
        min_width=min_width,
        max_depth=max_depth,
        current_depth=current_depth,
        execution_gaps=state.execution_gaps,
    )
    try:
        response = chain.invoke({"prompt": prompt})
        result = ReflectionResult.model_validate_json(response.content)
    except (LLMProviderError, LLMSchemaError) as exc:
        result = _fallback_reflection_from_quality(state.dag_quality_summary, exc)
    result = _apply_quality_override(result, state.dag_quality_summary)
    state.reflection_results.append(result)
    state.reflection_history.append(result.reason)
    state.status = "reflected"
    return state
