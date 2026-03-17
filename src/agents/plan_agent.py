"""Plan agent for the paper-oriented workflow front-end.

This agent translates the current signal context, DAG state, and reflection
history into an NVTA-style structured step plan.
"""

from __future__ import annotations

from typing import Any

from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import LLMClient
from src.operators import OperatorCatalog
from src.prompts import render_plan_prompt
from src.states import SignalContext, WorkflowState


def _dag_depth(state: WorkflowState) -> int:
    if not state.dag or not state.dag.nodes:
        return 0
    depth_by_node: dict[str, int] = {}
    for node in state.dag.nodes:
        if not node.parents:
            depth_by_node[node.node_id] = 1
        else:
            depth_by_node[node.node_id] = 1 + max(depth_by_node[parent] for parent in node.parents)
    return max(depth_by_node.values(), default=0)


def _build_signal_context(state: WorkflowState, protocol: DatasetProtocol) -> SignalContext:
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


def plan_agent(
    state: WorkflowState,
    protocol: DatasetProtocol,
    llm: LLMClient,
    catalog: OperatorCatalog,
) -> WorkflowState:
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
    state.step_plan = llm.generate_step_plan(
        prompt=prompt,
        instruction=state.user_instruction,
        signal_context=state.signal_context,
        dag_json=state.dag.model_dump() if state.dag else None,
        reflection=state.reflection_history,
        operator_catalog_summary=catalog.summary(),
    )
    state.status = "planned"
    return state
