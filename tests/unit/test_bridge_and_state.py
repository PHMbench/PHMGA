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


def _build_state(config_name: str) -> WorkflowState:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    graph_path = config["experiment"]["graph_path"]
    state = WorkflowState(user_instruction="paper workflow", dataset_name=protocol.dataset_name, graph_path=graph_path)
    state = plan_agent(state, protocol, llm)
    state = execute_agent(state, protocol, get_operator_catalog())
    state = reflect_agent(state, llm)
    return state


def test_workflow_state_is_serializable():
    state = _build_state("config/runs/rm101_synth_dag.yaml")
    payload = state.model_dump()
    assert payload["dataset_name"] == "RM101_SYNTH"
    assert payload["graph_path"] == "dag_only"
    assert payload["dag"]["nodes"]


def test_bridge_compiles_all_paths():
    for config_name, expected_graph_path in (
        ("config/runs/rm101_synth_dag.yaml", "dag_only"),
        ("config/runs/rm101_synth_ml.yaml", "ml"),
        ("config/runs/rm101_synth_torch.yaml", "torch"),
    ):
        state = _build_state(config_name)
        compiled = compile_dag_for_path(state.dag, expected_graph_path)
        assert compiled.manifest.path_type == expected_graph_path
        assert compiled.manifest.dag_hash
