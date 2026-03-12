from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent, reflect_agent
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _build_state(graph_path: str) -> WorkflowState:
    config = load_runtime_config(ROOT / "config/config.yaml", dataset_name="RM101", graph_path=graph_path)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    state = WorkflowState(user_instruction="paper workflow", dataset_name="RM101", graph_path=graph_path)
    state = plan_agent(state, protocol, llm)
    state = execute_agent(state, protocol, get_operator_catalog())
    state = reflect_agent(state, llm)
    return state


def test_workflow_state_is_serializable():
    state = _build_state("dag_only")
    payload = state.model_dump()
    assert payload["dataset_name"] == "RM101"
    assert payload["graph_path"] == "dag_only"
    assert payload["dag"]["nodes"]


def test_bridge_compiles_all_paths():
    for graph_path in ("dag_only", "ml", "torch"):
        state = _build_state(graph_path)
        compiled = compile_dag_for_path(state.dag, graph_path)
        assert compiled.manifest.path_type == graph_path
        assert compiled.manifest.dag_hash
