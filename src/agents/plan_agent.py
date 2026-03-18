"""Plan agent for the paper-oriented workflow front-end.

This agent translates the current signal context, DAG state, and reflection
history into an NVTA-style structured step plan.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.configuration import Configuration
from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import LLMClient
from src.model import LangChainLLMAdapter, get_llm
from src.operators import OperatorCatalog
from src.prompts import render_plan_prompt
from src.states import PHMState, SignalContext, StepPlan


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
    prompt = render_plan_prompt(
        instruction=state.user_instruction,
        signal_context=state.signal_context.model_dump(),
        dag_json=state.dag.model_dump() if state.dag else None,
        tools=catalog.summary(),
        reflection=state.reflection_history,
        current_depth=current_depth,
        min_depth=min_depth,
        min_width=min_width,
    )
    llm_adapter = _resolve_llm(state, llm)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm_adapter.bind_task(
        "plan",
        instruction=state.user_instruction,
        signal_context=state.signal_context,
        dag_json=state.dag.model_dump() if state.dag else None,
        reflection=state.reflection_history,
        operator_catalog_summary=catalog.summary(),
    )
    response = chain.invoke({"prompt": prompt})
    state.step_plan = StepPlan.model_validate(json.loads(response.content))
    state.status = "planned"
    return state
