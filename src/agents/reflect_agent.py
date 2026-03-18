"""Reflection agent for validating the current workflow state."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from src.configuration import Configuration
from src.llm import LLMClient
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
    response = chain.invoke({"prompt": prompt})
    result = ReflectionResult.model_validate(json.loads(response.content))
    state.reflection_results.append(result)
    state.reflection_history.append(result.reason)
    state.status = "reflected"
    return state
