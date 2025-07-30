from __future__ import annotations

import os
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from ..model import get_llm

from ..states.phm_states import PHMState, InputData
from ..schemas.plan_schema import AnalysisPlan
from ..prompts.plan_prompt import PLANNER_PROMPT 
from ..tools.signal_processing_schemas import OP_REGISTRY
import json


def _parse_plan(text: str) -> AnalysisPlan:
    """Convert raw numbered list text to an ``AnalysisPlan`` instance."""
    lines = [line.strip(" -") for line in text.splitlines() if line.strip()]
    return AnalysisPlan(steps=lines)


def get_available_nodes(state: PHMState) -> tuple:
    # Helper to categorize nodes for the prompt
    refs = [nid for nid, node in state.dag_state.nodes.items() if isinstance(node, InputData) and 'ref' in nid]
    tests = [nid for nid, node in state.dag_state.nodes.items() if isinstance(node, InputData) and 'test' in nid]
    others = [nid for nid, node in state.dag_state.nodes.items() if not isinstance(node, InputData)]
    return refs, tests, others

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

    # Prepare context for the prompt
    tools_schemas = json.dumps([cls.model_json_schema() for cls in OP_REGISTRY.values()], indent=2)
    ref_nodes, test_nodes, other_nodes = get_available_nodes(state)
    
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    
    try:
        structured_llm = llm.with_structured_output(AnalysisPlan)
    except NotImplementedError:
        structured_llm = None

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

