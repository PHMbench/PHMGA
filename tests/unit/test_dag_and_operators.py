from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.dag import validate_dag_json
from src.operators import get_operator_catalog
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def test_operator_catalog_declares_backend_availability_and_roles():
    catalog = get_operator_catalog()
    for spec in catalog.specs():
        assert spec.backend_availability
        assert spec.execution_role in {"trainable", "fixed", "proxy", "outer_only"}


def test_execute_agent_builds_validated_dag():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    state = WorkflowState(user_instruction="build dag", dataset_name=protocol.dataset_name, graph_path="ml")
    state = execute_agent(state, protocol, get_operator_catalog())
    dag = validate_dag_json(state.dag)
    assert len(dag.nodes) >= 8
    assert any(node.kind == "feature" for node in dag.nodes)
