from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm

from ..states.phm_states import PHMState
from ..utils import dag_to_llm_payload


def reflect_agent(state: PHMState) -> PHMState:
    """Evaluate the current DAG and update the revision flag.

    Parameters
    ----------
    state : PHMState
        State containing the analysis DAG.

    Returns
    -------
    PHMState
        Updated state with ``needs_revision`` and log entry appended.
    """
    llm = get_llm()
    dag_json = dag_to_llm_payload(state)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a PHM expert reviewing analysis steps."),
            ("human", "DAG: {dag}\nIs the analysis sufficient? Answer yes or no and explain."),
        ]
    )
    chain = prompt | llm
    resp = chain.invoke({"dag": dag_json})
    state.reflection_history.append(resp.content)
    state.needs_revision = "no" in resp.content.lower()
    return state

