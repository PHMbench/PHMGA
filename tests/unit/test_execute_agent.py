from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import StepPlan, WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _planned_state(config_name: str) -> tuple[WorkflowState, object, object, object]:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Execute the planned PHM pipeline.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8},
    )
    state = plan_agent(state, protocol, llm, catalog)
    return state, protocol, llm, catalog


def test_execute_agent_materializes_results_and_multi_node():
    state, protocol, llm, catalog = _planned_state("config/runs/rm101_synth_ml.yaml")
    state = execute_agent(state, protocol, catalog, llm)

    assert "ch1" in state.execution_results
    assert any(node.kind == "multi" for node in state.dag.nodes)
    for node in state.dag.nodes:
        if node.kind == "input":
            continue
        assert node.plan_step_ref
        assert node.rationale


def test_execute_agent_records_unknown_operator_gap_without_silent_fallback():
    state, protocol, llm, catalog = _planned_state("config/runs/rm101_synth_dag.yaml")
    state.step_plan = StepPlan.model_validate(
        {
            "plan": [
                {"parent": "ch1", "op_name": "nonexistent_op", "params": {}},
            ]
        }
    )

    state = execute_agent(state, protocol, catalog, llm)

    assert state.execution_gaps
    assert "Unknown or unsupported operator" in state.execution_gaps[0].message
    assert all(node.kind == "input" for node in state.dag.nodes)
