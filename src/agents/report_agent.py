"""Report agent that turns graph-dependent artifacts into final prose."""

from __future__ import annotations

from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

from src.bridge import CompiledDagManifest
from src.configuration import Configuration
from src.data import DatasetProtocol
from src.llm import LLMClient
from src.model import LangChainLLMAdapter, get_llm
from src.prompts import render_report_prompt
from src.states import PHMState


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


def report_agent(
    state: PHMState,
    protocol: DatasetProtocol,
    manifest: CompiledDagManifest,
    path_artifacts: Dict[str, Any],
    llm: LLMClient | None = None,
) -> str:
    """Build the final markdown report from artifacts plus reflection context."""

    review_context = {
        "instruction": state.user_instruction,
        "stage": str(state.data_context.get("stage", "FINAL_REPORT")),
        "dag_blueprint": state.dag.model_dump() if state.dag else {"nodes": [], "edges": []},
        "issues_summary": "\n".join(gap.message for gap in state.execution_gaps),
        "min_depth": int(state.data_context.get("min_depth", 2)),
        "min_width": int(state.data_context.get("min_width", 1)),
        "max_depth": int(state.data_context.get("max_depth", 8)),
        "current_depth": _dag_depth(state),
        "round_count": len(state.round_history),
    }
    reflection_summary = (
        state.reflection_results[-1].model_dump()
        if state.reflection_results
        else {"decision": "unknown", "reason": "No reflection was recorded."}
    )
    prompt = render_report_prompt(
        instruction=state.user_instruction,
        graph_path=state.graph_path,
        compiled_manifest=manifest.model_dump(),
        path_artifacts=path_artifacts,
        reflection_summary=reflection_summary,
        dag_quality_summary=state.dag_quality_summary,
        review_context=review_context,
    )
    state.status = "reported"
    llm_adapter = _resolve_llm(state, llm)
    chain = ChatPromptTemplate.from_template("{prompt}") | llm_adapter.bind_task(
        "report",
        instruction=state.user_instruction,
        dataset_name=protocol.dataset_name,
        graph_path=state.graph_path,
        compiled_manifest=manifest.model_dump(),
        path_artifacts=path_artifacts,
        reflection_summary=reflection_summary,
        dag_quality_summary=state.dag_quality_summary,
        review_context=review_context,
        step_plan=state.step_plan.model_dump() if state.step_plan else {"plan": []},
    )
    response = chain.invoke({"prompt": prompt})
    return response.content
