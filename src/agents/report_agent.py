"""Report agent that turns graph-dependent artifacts into final prose."""

from __future__ import annotations

from typing import Any, Dict

from src.bridge import CompiledDagManifest
from src.data import DatasetProtocol
from src.llm import OfflineLLM
from src.prompts import render_report_prompt
from src.states import WorkflowState


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


def report_agent(
    state: WorkflowState,
    protocol: DatasetProtocol,
    manifest: CompiledDagManifest,
    path_artifacts: Dict[str, Any],
    llm: OfflineLLM,
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
    render_report_prompt(
        instruction=state.user_instruction,
        graph_path=state.graph_path,
        compiled_manifest=manifest.model_dump(),
        path_artifacts=path_artifacts,
        reflection_summary=reflection_summary,
        review_context=review_context,
    )
    state.status = "reported"
    return llm.render_report(
        instruction=state.user_instruction,
        dataset_name=protocol.dataset_name,
        graph_path=state.graph_path,
        compiled_manifest=manifest.model_dump(),
        path_artifacts=path_artifacts,
        reflection_summary=reflection_summary,
        review_context=review_context,
        step_plan=state.step_plan.model_dump() if state.step_plan else {"plan": []},
    )
