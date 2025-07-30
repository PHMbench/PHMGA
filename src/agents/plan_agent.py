from __future__ import annotations

import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm

from ..states.phm_states import PHMState
from ..schemas.plan_schema import AnalysisPlan


def _parse_plan(text: str) -> AnalysisPlan:
    """Convert raw numbered list text to an ``AnalysisPlan`` instance."""
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    return AnalysisPlan(steps=lines)


def plan_agent(state: PHMState) -> PHMState:
    """Generate a high-level analysis plan from the user instruction.

    Parameters
    ----------
    state : PHMState
        State containing the user's initial request.

    Returns
    -------
    PHMState
        Updated state with ``analysis_plan`` and ``high_level_plan`` filled.
    """
    llm = get_llm()
    try:
        structured_llm = llm.with_structured_output(AnalysisPlan)
    except NotImplementedError:
        structured_llm = None
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a goal decomposition assistant."),
            (
                "human",
                "User instruction: {instruction}\nReturn the plan as a numbered list",
            ),
        ]
    )
    if structured_llm:
        chain = prompt | structured_llm
        plan = chain.invoke({"instruction": state.user_instruction})
    else:
        chain = prompt | llm
        resp = chain.invoke({"instruction": state.user_instruction})
        plan = _parse_plan(resp.content)
    state.analysis_plan = plan
    state.high_level_plan = plan.steps
    return state

