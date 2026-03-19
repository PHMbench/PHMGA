from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent, reflect_agent, report_agent
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.evaluation import build_dag_quality_summary
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _reflected_state(config_name: str) -> tuple[WorkflowState, object, object]:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Write the final PHM report.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "FINAL_REPORT"},
    )
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    state.dag_quality_summary = build_dag_quality_summary(state, protocol, config, catalog).model_dump()
    state = reflect_agent(state, llm)
    return state, protocol, llm


def test_report_agent_writes_dag_only_sections():
    state, protocol, llm = _reflected_state("config/runs/rm101_synth_dag.yaml")
    compiled = compile_dag_for_path(state.dag, "dag_only")
    report = report_agent(
        state,
        protocol,
        compiled.manifest,
        {
            "node_inventory": compiled.node_inventory,
            "edge_inventory": compiled.edge_inventory,
            "method_description": compiled.method_description,
            "decision_side_outputs": {"threshold_01": {"decision": True, "score": 0.9}},
        },
        llm,
    )
    assert "## DAG Evidence" in report
    assert "## DAG Quality" in report
    assert "## Dataset-Level Diagnosis Evidence" in report
    assert "## Decision Side Outputs" in report
    assert "Reflection decision" in report


def test_report_agent_writes_ml_and_torch_sections():
    ml_state, ml_protocol, llm = _reflected_state("config/runs/rm101_synth_ml.yaml")
    ml_manifest = compile_dag_for_path(ml_state.dag, "ml").manifest
    ml_report = report_agent(
        ml_state,
        ml_protocol,
        ml_manifest,
        {
            "feature_pipeline": {
                "execution_nodes": [{"node_id": "ch1"}],
                "output_specs": [{"output_node_id": "n1", "output_kind": "feature"}],
                "output_policy": "terminal_only",
            },
            "metrics": {"test": {"accuracy": 0.8, "macro_f1": 0.7}},
            "importance": {"feature.rms": 0.5},
        },
        llm,
    )
    assert "## ML Evidence" in ml_report
    assert "## DAG Quality" in ml_report
    assert "## Dataset-Level Diagnosis Evidence" in ml_report

    torch_state, torch_protocol, llm = _reflected_state("config/runs/rm101_synth_torch.yaml")
    torch_manifest = compile_dag_for_path(torch_state.dag, "torch").manifest
    torch_report = report_agent(
        torch_state,
        torch_protocol,
        torch_manifest,
        {
            "model_build_plan": {"backend_target": "torch"},
            "training_curves": {"loss": [1.0, 0.5]},
            "checkpoint": {"epoch": 1},
        },
        llm,
    )
    assert "## Torch Evidence" in torch_report
    assert "## DAG Quality" in torch_report
    assert "## Dataset-Level Diagnosis Evidence" in torch_report
