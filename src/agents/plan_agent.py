from __future__ import annotations

import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from ..model import get_default_llm

from ..states.phm_states import PHMState


def plan_agent(state: PHMState) -> PHMState:
    """Generate a list of coarse analysis steps from the user instruction.

    Parameters
    ----------
    state : PHMState
        State object containing the original user instruction.

    Returns
    -------
    PHMState
        The state with ``high_level_plan`` populated.
    """
    llm = get_default_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a goal decomposition assistant."),
            ("human", "User instruction: {instruction}\nReturn each step as a list"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"instruction": state.user_instruction})
    lines = [line.strip(" -") for line in result.content.splitlines() if line.strip()]
    state.high_level_plan = lines
    return state

