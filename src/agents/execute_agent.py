from __future__ import annotations

import json
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from ..model import get_default_llm

from ..states.phm_states import PHMState
from ..tools.signal_processing_schemas import OP_REGISTRY, get_operator
from ..utils import dag_to_llm_payload


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
    llm = get_default_llm()

    signal = np.asarray(state.test_signal.data.get("signal", []))
    tracker = state.tracker()

    for _ in range(MAX_STEPS):
        dag_json = dag_to_llm_payload(state)
        tools_desc = "\n".join(
            f"- {name}: {cls.description}" for name, cls in OP_REGISTRY.items()
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Select the next best operation in JSON format given the plan, current DAG and tools.",
                ),
                ("human", "Plan: {plan}\nDAG: {dag}\nTools:\n{tools}\n"),
            ]
        )
        chain = prompt | llm
        resp = chain.invoke({"plan": "\n".join(state.high_level_plan), "dag": dag_json, "tools": tools_desc})
        try:
            spec = json.loads(resp.content)
        except Exception:
            break
        if spec.get("op_name") == "stop":
            break
        op_cls = get_operator(spec["op_name"])
        op = op_cls(**spec.get("params", {}), parent=state.dag_state.leaves)
        signal = op.execute(signal)
        tracker.add_execution(op)

    return state

