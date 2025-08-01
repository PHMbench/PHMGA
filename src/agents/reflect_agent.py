from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm
from ..configuration import Configuration
from ..states.phm_states import PHMState
from ..prompts.reflect_prompt import REFLECT_PROMPT


def reflect_agent(state: PHMState, *, stage: str) -> dict:
    """Assess the current analysis stage and decide if revision is needed."""

    llm = get_llm(Configuration())
    if stage == "DAG_REVIEW":
        summary = state.tracker().export_json()
    else:
        summary = "\n".join(
            f"- {ins.content}" for ins in state.insights
        ) or "No insights."

    prompt = ChatPromptTemplate.from_template(REFLECT_PROMPT)
    chain = prompt | llm
    resp = chain.invoke({
        "instruction": state.user_instruction,
        "stage": stage,
        "summary": summary,
    })
    try:
        data = json.loads(resp.content)
    except Exception as e:
        data = {"is_sufficient": False, "reason": f"parse error: {e}"}

    needs_revision = not data.get("is_sufficient", False)
    history = state.reflection_history + [data.get("reason", "")]
    return {"needs_revision": needs_revision, "reflection_history": history}


if __name__ == "__main__":
    import numpy as np
    from ..states.phm_states import DAGState, InputData

    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="diagnose",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
    )
    out = reflect_agent(state, stage="DAG_REVIEW")
    print(out)
