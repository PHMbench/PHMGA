from __future__ import annotations

import json
from langchain_core.prompts import ChatPromptTemplate

from ..model import get_llm
from ..states.phm_states import PHMState
from ..prompts.plan_prompt import PLANNER_PROMPT
from ..tools.signal_processing_schemas import OP_REGISTRY


def plan_agent(state: PHMState) -> dict:
    """Call LLM to generate a detailed processing plan."""

    llm = get_llm()
    tools_schemas = json.dumps([cls.model_json_schema() for cls in OP_REGISTRY.values()], ensure_ascii=False)
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    chain = prompt | llm
    resp = chain.invoke(
        {
            "instruction": state.user_instruction,
            "reference_root": state.dag_state.reference_root,
            "test_root": state.dag_state.test_root,
            "tools": tools_schemas,
        }
    )
    try:
        plan_json = json.loads(resp.content)
        detailed_plan = plan_json.get("plan", plan_json)
    except Exception as e:
        state.error_logs.append(f"Planner parse error: {e}")
        detailed_plan = []

    return {"detailed_plan": detailed_plan}


if __name__ == "__main__":
    import numpy as np
    from ..states.phm_states import DAGState, InputData

    ref = InputData(node_id="ref", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    test = InputData(node_id="test", data={"signal": np.ones(4)}, parents=[], shape=(4,))
    dag = DAGState(user_instruction="demo", reference_root="ref", test_root="test")
    state = PHMState(
        user_instruction="demo instruction",
        reference_signal=ref,
        test_signal=test,
        dag_state=dag,
    )
    print("Initial state:", state.model_dump())
    result = plan_agent(state)
    print("Planner output:", result)

