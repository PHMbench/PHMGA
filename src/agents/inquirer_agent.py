from __future__ import annotations

"""Agent responsible for generating analysis insights by comparing DAG nodes."""

from typing import List, Tuple

from ..model import get_llm
from ..states.phm_states import PHMState
from ..tools.comparator_tool import compare_processed_nodes
from ..schemas.insight_schema import AnalysisInsight


def find_comparable_node_pairs(state: PHMState) -> List[Tuple[str, str]]:
    """Return node id pairs that should be compared.

    Parameters
    ----------
    state : PHMState
        Current pipeline state containing the DAG structure.

    Returns
    -------
    List[Tuple[str, str]]
        Pairs of node identifiers representing reference and test nodes.
    """

    # Minimal logic: take the first two leaves (reference, test)
    if len(state.dag_state.leaves) < 2:
        return []
    ref = state.dag_state.leaves[0]
    test = state.dag_state.leaves[1]
    return [(ref, test)]


def inquirer_agent(state: PHMState) -> dict:
    """Generate insights by comparing nodes in the DAG.

    Parameters
    ----------
    state : PHMState
        The full analysis state including the DAG and available nodes.

    Returns
    -------
    dict
        Dictionary with a single key ``"insights"`` containing a list of
        :class:`AnalysisInsight` objects.
    """

    llm = get_llm()
    try:
        llm_with_tools = llm.bind_tools([compare_processed_nodes])
    except NotImplementedError:
        llm_with_tools = None

    pairs = find_comparable_node_pairs(state)
    if not pairs:
        return {"insights": []}

    prompt = (
        "Based on the current DAG, compare the following node pairs: " f"{pairs}"
    )
    ai_msg = llm_with_tools.invoke(prompt) if llm_with_tools else None

    new_insights: List[AnalysisInsight] = []
    if ai_msg and getattr(ai_msg, "tool_calls", None):
        for call in ai_msg.tool_calls:
            args = call.get("args", {})
            args["state"] = state
            insight = compare_processed_nodes(**args)
            new_insights.append(insight)
    else:
        # Fallback for simple mock LLMs
        for ref_id, test_id in pairs:
            insight = compare_processed_nodes(
                state=state, reference_node_id=ref_id, test_node_id=test_id
            )
            new_insights.append(insight)

    return {"insights": new_insights}
