from __future__ import annotations


from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm

from ..states.phm_states import PHMState


def report_agent(state: PHMState) -> PHMState:
    """Generate the final markdown report and DAG image.

    Parameters
    ----------
    state : PHMState
        State after all analysis steps have been executed.

    Returns
    -------
    PHMState
        The state populated with ``final_report`` and a PNG graph file.
    """
    tracker = state.tracker()
    try:
        tracker.write_png("final_dag.png")
    except Exception:
        pass
    table = "| Insight | Severity | Nodes |\n|---|---|---|\n"
    for ins in state.insights:
        table += (
            f"| {ins.content} | {ins.severity_score:.2f} | `{ins.compared_nodes[0]}` ↔ `{ins.compared_nodes[1]}` |\n"
        )
    state.final_report = f"# PHM Report\n\n## Insights\n{table}\n\n## DAG\n![](final_dag.png)"
    return state


if __name__ == "__main__":
    import numpy as np
    from ..states.phm_states import DAGState, InputData

    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="demo",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
    )
    result = report_agent(state)
    print(result.final_report)

