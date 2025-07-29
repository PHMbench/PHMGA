from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from ..model import get_default_llm

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
    llm = get_default_llm()
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
    # Mark revision required if the LLM does not confirm sufficiency
    lower = resp.content.lower()
    state.needs_revision = "yes" not in lower and "sufficient" not in lower
    return state

