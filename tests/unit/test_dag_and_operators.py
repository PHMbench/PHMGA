from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.dag import validate_dag_json
from src.llm import get_llm
from src.operators import (
    get_aggregate_operators,
    get_decision_operators,
    get_expand_operators,
    get_multi_operators,
    get_operator_catalog,
    get_transform_operators,
)
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def test_operator_catalog_declares_backend_availability_and_roles():
    catalog = get_operator_catalog()
    for spec in catalog.specs():
        assert spec.backend_availability
        assert "pt" in spec.backend_availability
        assert spec.execution_role in {"trainable", "fixed", "proxy", "outer_only"}
        assert spec.schema_category in {"EXPAND", "TRANSFORM", "AGGREGATE", "MULTI_VARIABLE", "DECISION"}
        assert spec.rank_class in {"rank_up", "rank_same", "rank_down", "multi_input", "terminal_decision"}
        assert spec.description
        assert spec.legal_paths
        assert spec.op_name
        assert spec.input_spec
        assert spec.output_spec
    assert catalog.resolve_plan_name("fft") == "signal.fft_mag"
    assert catalog.resolve_plan_name("concatenate") == "multi.concatenate"
    assert catalog.resolve_plan_name("stft") == "signal.stft"
    assert catalog.resolve_plan_name("threshold") == "decision.threshold"
    summary = catalog.summary()
    assert all("schema_category" in item for item in summary)
    assert all("rank_class" in item for item in summary)
    assert all("description" in item for item in summary)
    assert all("input_spec" in item and "output_spec" in item for item in summary)
    assert any(item["op_name"] == "concatenate" and item["llm_tunable_params"] for item in summary)
    assert any(item["schema_category"] == "EXPAND" for item in summary)
    assert any(item["schema_category"] == "DECISION" for item in summary)


def test_operator_modules_return_expected_categories_and_catalog_is_assembly_only():
    groups = {
        "EXPAND": get_expand_operators(),
        "TRANSFORM": get_transform_operators(),
        "AGGREGATE": get_aggregate_operators(),
        "MULTI_VARIABLE": get_multi_operators(),
        "DECISION": get_decision_operators(),
    }
    for category, operators in groups.items():
        assert operators
        assert all(operator.spec.schema_category == category for operator in operators)

    catalog_source = (ROOT / "src/operators/catalog.py").read_text(encoding="utf-8")
    assert "class NormalizeOperator" not in catalog_source
    assert "class ThresholdDecisionOperator" not in catalog_source
    assert "get_expand_operators" in catalog_source
    assert "get_transform_operators" in catalog_source
    assert "get_aggregate_operators" in catalog_source
    assert "get_multi_operators" in catalog_source
    assert "get_decision_operators" in catalog_source


def test_execute_agent_builds_validated_dag():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="build dag",
        dataset_name=protocol.dataset_name,
        graph_path="ml",
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8},
    )
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    dag = validate_dag_json(state.dag)
    assert len(dag.nodes) >= 8
    assert any(node.kind == "feature" for node in dag.nodes)
    assert any(node.kind == "multi" for node in dag.nodes)
    assert any(node.kind == "decision" for node in dag.nodes)
    assert any(node.operator_category == "MULTI_VARIABLE" for node in dag.nodes if node.kind == "multi")
    assert any(node.operator_category == "EXPAND" for node in dag.nodes)
