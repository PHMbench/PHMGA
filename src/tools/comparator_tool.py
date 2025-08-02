from __future__ import annotations

from typing import Any
import numpy as np

from ..states.phm_states import PHMState, ProcessedData, InputData
from ..schemas.insight_schema import AnalysisInsight


def compare_processed_nodes(
    state: PHMState,
    reference_node_id: str,
    test_node_id: str,
) -> AnalysisInsight:
    """Compare data at two nodes and produce an ``AnalysisInsight``.

    Parameters
    ----------
    state : PHMState
        Current state containing the DAG and node data.
    reference_node_id : str
        Node id representing the reference signal.
    test_node_id : str
        Node id representing the signal under test.

    Returns
    -------
    AnalysisInsight
        Structured insight describing the difference between the two nodes.
    """
    ref_node = state.dag_state.nodes.get(reference_node_id)
    test_node = state.dag_state.nodes.get(test_node_id)
    if isinstance(ref_node, ProcessedData):
        ref = np.asarray(ref_node.processed_data)
    elif isinstance(ref_node, InputData):
        ref = np.asarray(ref_node.data.get("signal", []))
    else:
        raise ValueError("Invalid reference node")

    if isinstance(test_node, ProcessedData):
        test = np.asarray(test_node.processed_data)
    elif isinstance(test_node, InputData):
        test = np.asarray(test_node.data.get("signal", []))
    else:
        raise ValueError("Invalid test node")
    diff = float(np.mean(np.abs(ref - test)))
    severity = min(1.0, diff / (np.mean(np.abs(ref)) + 1e-6))

    content = f"Difference between {reference_node_id} and {test_node_id} is {diff:.3f}"
    insight = AnalysisInsight(
        content=content,
        severity_score=severity,
        compared_nodes=(reference_node_id, test_node_id),
    )
    return insight


if __name__ == "__main__":
    print("--- Testing comparator_tool.py ---")

    # Create simple reference and test signals
    ref_signal = np.array([1.0, 2.0, 3.0])
    test_signal = np.array([1.5, 2.5, 3.5])

    # Wrap signals in InputData nodes
    ref_node = InputData(node_id="ref", parents=[], shape=ref_signal.shape, data={"signal": ref_signal})
    test_node = InputData(node_id="test", parents=[], shape=test_signal.shape, data={"signal": test_signal})

    # Build a minimal PHMState with these nodes
    from ..states.phm_states import DAGState

    dag_state = DAGState(
        user_instruction="",
        reference_root="ref",
        test_root="test",
        nodes={"ref": ref_node, "test": test_node},
        leaves=["ref", "test"],
    )

    state = PHMState(
        user_instruction="",
        reference_signal=ref_node,
        test_signal=test_node,
        dag_state=dag_state,
    )

    # Execute comparison
    insight = compare_processed_nodes(state, "ref", "test")
    print(insight)
    assert insight.compared_nodes == ("ref", "test")
    assert insight.severity_score >= 0

    print("\n--- comparator_tool.py tests passed! ---")
