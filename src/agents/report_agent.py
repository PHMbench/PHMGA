from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from ..model import get_default_llm

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
    tracker.write_png("final_dag")
    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write a concise PHM report."),
            ("human", "Plan: {plan}"),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"plan": "\n".join(state.high_level_plan)})
    state.final_report = resp.content + "\n\n![](final_dag.png)"
    return state

