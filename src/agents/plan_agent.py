from __future__ import annotations

from src.data import DatasetProtocol
from src.llm import OfflineLLM
from src.states import WorkflowState


def plan_agent(state: WorkflowState, protocol: DatasetProtocol, llm: OfflineLLM) -> WorkflowState:
    state.plan = llm.generate_plan(protocol.dataset_name, state.graph_path)
    state.status = "planned"
    return state
