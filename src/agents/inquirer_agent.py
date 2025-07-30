from __future__ import annotations

"""Agent responsible for generating analysis insights."""

from typing import List, Tuple

from ..model import get_llm
from ..states.phm_states import PHMState
from ..tools.comparator_tool import compare_processed_nodes
from ..schemas.insight_schema import AnalysisInsight


def find_comparable_node_pairs(state: PHMState) -> List[Tuple[str, str]]:
    """Return candidate node id pairs for comparison.

    Parameters
    ----------
    state : PHMState
        Current pipeline state containing the DAG information.

    Returns
    -------
    list[tuple[str, str]]
        List of `(reference_node_id, test_node_id)` pairs. The current
        implementation simply returns the first two leaves of the DAG if
        present.
    """

    if len(state.dag_state.leaves) < 2:
        return []
    ref = state.dag_state.leaves[0]
    test = state.dag_state.leaves[1]
    return [(ref, test)]


def _run_comparisons(state: PHMState, pairs: List[Tuple[str, str]]) -> List[AnalysisInsight]:
    """Execute comparison tool for each provided pair."""
    insights: List[AnalysisInsight] = []
    for ref_id, test_id in pairs:
        insights.append(
            compare_processed_nodes(state=state, reference_node_id=ref_id, test_node_id=test_id)
        )
    return insights


def inquirer_agent(state: PHMState) -> dict:
    """Review the DAG and generate insights by comparing node outputs."""
    llm = get_llm()
    try:
        llm_with_tools = llm.bind_tools([compare_processed_nodes])
    except NotImplementedError:
        llm_with_tools = None

    pairs = find_comparable_node_pairs(state)
    if not pairs:
        return {"insights": []}

    prompt = f"Based on the current DAG, compare the following node pairs: {pairs}"
    ai_msg = llm_with_tools.invoke(prompt) if llm_with_tools else None

    if ai_msg and getattr(ai_msg, "tool_calls", None):
        insights = []
        for call in ai_msg.tool_calls:
            args = call.get("args", {})
            args["state"] = state
            insights.append(compare_processed_nodes(**args))
    else:
        # Fallback for simple or mock LLMs
        insights = _run_comparisons(state, pairs)

    return {"insights": insights}
