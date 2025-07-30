"""Reporter agent that assembles the final PHM report."""

from __future__ import annotations

from ..states.phm_states import PHMState


def _build_insight_table(state: PHMState) -> str:
    """Create a Markdown table summarizing all insights."""

    header = "| Insight | Severity | Nodes |\n|---|---|---|\n"
    rows = [
        f"| {ins.content} | {ins.severity_score:.2f} | `{ins.compared_nodes[0]}` ↔ `{ins.compared_nodes[1]}` |"
        for ins in state.insights
    ]
    return header + "\n".join(rows) + "\n"


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
        # Visualization failure should not halt report generation
        pass

    table = _build_insight_table(state)
    state.final_report = f"# PHM Report\n\n## Insights\n{table}\n\n## DAG\n![](final_dag.png)"
    return state

