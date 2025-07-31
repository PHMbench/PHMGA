from __future__ import annotations

import json
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm

from ..states.phm_states import PHMState
from ..tools.signal_processing_schemas import OP_REGISTRY, get_operator
from ..utils import dag_to_llm_payload
from ..prompts.execute_prompt import EXECUTE_PROMPT

MAX_STEPS = 10


def execute_agent(state: PHMState) -> PHMState:
    """Iteratively execute operators chosen by an LLM.

    Parameters
    ----------
    state : PHMState
        Current pipeline state containing the DAG and high level plan.

    Returns
    -------
    PHMState
        The updated state with a richer inner DAG.
    """
    llm = get_llm()

    signal = np.asarray(state.test_signal.data.get("signal", []))

    tracker = state.tracker()

    tools_desc = "\n".join(
        f"- {name}: {cls.description}" for name, cls in OP_REGISTRY.items()
    )

    # 智能决策 Prompt
    prompt_template = ChatPromptTemplate.from_template(
        EXECUTE_PROMPT
    )
    chain = prompt_template | llm
    for _ in range(MAX_STEPS):
        dag_json = dag_to_llm_payload(state)


        resp = chain.invoke({"plan": "\n".join(state.high_level_plan),
                              "dag": dag_json,
                              "leaves": state.dag_state.leaves,
                              "tools": tools_desc})
        try:
            spec = json.loads(resp.content)
        except Exception:
            print(f"Warning: LLM returned invalid JSON: {resp.content}")
            continue # or break
        if spec.get("op_name") == "stop":
            break

        
        op_cls = get_operator(spec["op_name"])
        op = op_cls(**spec.get("params", {}), parent=state.dag_state.leaves)
        signal = op.execute(signal)
        tracker.add_execution(op)

    return state

