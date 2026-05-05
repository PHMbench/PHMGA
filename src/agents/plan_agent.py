"""Plan agent for the paper-oriented workflow front-end.

This agent translates the current signal context, DAG state, and reflection
history into an NVTA-style structured step plan.
"""

from __future__ import annotations

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from src.configuration import Configuration
from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import LLMClient, LLMProviderError, LLMSchemaError
from src.model import LangChainLLMAdapter, get_llm
from src.operators import OperatorCatalog
from src.prompts import render_plan_prompt, render_supervisor_proving_plan_prompt
from src.states import PHMState, PlanStep, SignalContext, StepPlan


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


def _build_signal_context(state: PHMState, protocol: DatasetProtocol) -> SignalContext:
    sample_id, preview_window = materialize_preview_signal(protocol)
    channel_count = int(preview_window.shape[0])
    return SignalContext(
        dataset_name=protocol.dataset_name,
        channel_count=channel_count,
        window_shape=list(preview_window.shape),
        sampling_rate=int(protocol.samples[0].sampling_rate),
        source_mode=protocol.source_mode,
        root_node_ids=[f"ch{index + 1}" for index in range(channel_count)],
        representative_sample_id=sample_id,
    )


def _resolve_llm(state: PHMState, llm: LLMClient | None) -> LangChainLLMAdapter:
    if llm is not None:
        return LangChainLLMAdapter(llm)
    return get_llm(Configuration.from_runtime_config(state.runtime_config))


def _planner_trace_context(state: PHMState) -> dict[str, Any]:
    runtime_cfg = dict(state.runtime_config.get("runtime", {}))
    return {
        "output_dir": runtime_cfg.get("output_dir", ""),
        "config_name": runtime_cfg.get("config_name", ""),
        "dataset_name": state.dataset_name,
        "graph_path": state.graph_path,
    }


def _workflow_mode(state: PHMState) -> str:
    return str(state.runtime_config.get("runtime", {}).get("workflow_mode", "rich"))


def _parent_slug(parent: str) -> str:
    return parent.replace(",", "__")


def _fallback_node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{_parent_slug(parent)}"


def _aggregate_ops(catalog: OperatorCatalog) -> list[str]:
    candidates = ["rms", "mean", "std", "kurtosis", "crest_factor"]
    valid: list[str] = []
    for op_name in candidates:
        try:
            catalog.get_by_plan_name(op_name)
        except KeyError:
            continue
        valid.append(op_name)
    return valid


def _existing_feature_parent_ids(state: PHMState, min_width: int) -> list[str]:
    if not state.dag:
        return []
    preferred: list[str] = []
    fallback: list[str] = []
    for node in state.dag.nodes:
        if node.kind == "transform":
            preferred.append(node.node_id)
        elif node.kind == "input":
            fallback.append(node.node_id)
    parent_ids = preferred or fallback
    limit = max(min_width, 1)
    return parent_ids[:limit]


def _dag_has_feature_nodes(state: PHMState) -> bool:
    return bool(state.dag and any(node.kind == "feature" for node in state.dag.nodes))


def _plan_has_aggregate_steps(plan: StepPlan, catalog: OperatorCatalog) -> bool:
    for step in plan.plan:
        try:
            operator = catalog.get_by_plan_name(step.op_name)
        except KeyError:
            continue
        if operator.spec.schema_category == "AGGREGATE":
            return True
    return False


def _deterministic_feature_plan(state: PHMState, catalog: OperatorCatalog, min_width: int) -> StepPlan:
    """Build a locally executable feature plan when the provider is unavailable."""

    aggregate_ops = _aggregate_ops(catalog)
    if not aggregate_ops:
        raise LLMProviderError("deterministic planner fallback requires at least one aggregate operator")

    steps: list[PlanStep] = []
    existing_feature_parents = _existing_feature_parent_ids(state, min_width)
    if existing_feature_parents:
        for parent in existing_feature_parents:
            for op_name in aggregate_ops:
                steps.append(PlanStep(parent=parent, op_name=op_name, params={}))
        return StepPlan(plan=steps)

    root_ids = list(state.signal_context.root_node_ids if state.signal_context else [])
    if not root_ids:
        raise LLMProviderError("deterministic planner fallback requires signal root ids")

    for root_id in root_ids:
        normalize_step_index = len(steps) + 1
        steps.append(PlanStep(parent=root_id, op_name="normalize", params={"eps": 1e-6}))
        normalize_id = _fallback_node_id(normalize_step_index, "normalize", root_id)

        fft_step_index = len(steps) + 1
        steps.append(PlanStep(parent=normalize_id, op_name="fft", params={}))
        fft_id = _fallback_node_id(fft_step_index, "fft", normalize_id)

        for op_name in aggregate_ops:
            steps.append(PlanStep(parent=fft_id, op_name=op_name, params={}))

    return StepPlan(plan=steps)


def _ensure_model_path_feature_plan(
    state: PHMState,
    catalog: OperatorCatalog,
    min_width: int,
    plan: StepPlan,
) -> StepPlan:
    if state.graph_path not in {"ml", "torch"}:
        return plan
    if _dag_has_feature_nodes(state) or _plan_has_aggregate_steps(plan, catalog):
        return plan
    state.reflection_history.append(
        "deterministic_feature_plan_guard: provider plan lacked aggregate feature outputs for model path"
    )
    return _deterministic_feature_plan(state, catalog, min_width)


def _fallback_plan_from_provider_error(
    state: PHMState,
    catalog: OperatorCatalog,
    min_width: int,
    exc: LLMProviderError | LLMSchemaError | ValidationError,
) -> StepPlan:
    if _workflow_mode(state) == "supervisor_proving":
        raise exc
    plan = _deterministic_feature_plan(state, catalog, min_width)
    state.reflection_history.append(
        f"provider_plan_fallback: {type(exc).__name__}; generated deterministic normalize/fft/feature plan"
    )
    return plan


def plan_agent(
    state: PHMState,
    protocol: DatasetProtocol,
    llm: LLMClient | None,
    catalog: OperatorCatalog,
) -> PHMState:
    """Generate a structured step plan from signal context and the current DAG."""

    if state.signal_context is None:
        state.signal_context = _build_signal_context(state, protocol)

    current_depth = _dag_depth(state)
    min_depth = int(state.data_context.get("min_depth", 2))
    min_width = int(state.data_context.get("min_width", 1))
    prompt_args = {
        "instruction": state.user_instruction,
        "signal_context": state.signal_context.model_dump(),
        "dag_json": state.dag.model_dump() if state.dag else None,
        "tools": catalog.summary(),
        "reflection": state.reflection_history,
        "current_depth": current_depth,
        "min_depth": min_depth,
        "min_width": min_width,
    }
    if _workflow_mode(state) == "supervisor_proving":
        prompt = render_supervisor_proving_plan_prompt(**prompt_args)
    else:
        prompt = render_plan_prompt(**prompt_args)
    llm_adapter = _resolve_llm(state, llm)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm_adapter.bind_task(
        "plan",
        instruction=state.user_instruction,
        signal_context=state.signal_context,
        dag_json=state.dag.model_dump() if state.dag else None,
        reflection=state.reflection_history,
        operator_catalog_summary=catalog.summary(),
        trace_context=_planner_trace_context(state),
    )
    try:
        response = chain.invoke({"prompt": prompt})
        state.step_plan = StepPlan.model_validate_json(response.content)
    except (LLMProviderError, LLMSchemaError, ValidationError) as exc:
        state.step_plan = _fallback_plan_from_provider_error(state, catalog, min_width, exc)
    state.step_plan = _ensure_model_path_feature_plan(state, catalog, min_width, state.step_plan)
    state.status = "planned"
    return state
