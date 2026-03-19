from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.evaluation import build_dag_quality_summary
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _executed_state(config_name: str) -> tuple[WorkflowState, object, object, dict]:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Evaluate the current DAG quality.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "POST_EXECUTE"},
    )
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    return state, protocol, catalog, config


def test_dag_quality_summary_builds_without_proxy_probe():
    state, protocol, catalog, config = _executed_state("config/runs/rm101_synth_dag.yaml")
    summary = build_dag_quality_summary(state, protocol, config, catalog)

    assert summary.current_depth >= 1
    assert summary.feature_node_count >= 1
    assert summary.execution_gap_count == len(state.execution_gaps)
    assert 0.0 <= summary.nan_ratio <= 1.0
    assert 0.0 <= summary.zero_variance_ratio <= 1.0
    assert summary.proxy_probe_enabled is False
    assert summary.proxy_probe_macro_f1 is None
    assert summary.dataset_level["enabled"] is True
    assert summary.dataset_level["source"] == "split_subset_execution"
    assert "train" in summary.dataset_level["split_window_counts"]
    assert "materialization_ok" in summary.dataset_level
    assert "decision_summary" in summary.dataset_level


def test_dag_quality_summary_can_run_proxy_probe_when_enabled():
    state, protocol, catalog, config = _executed_state("config/runs/rm101_synth_ml.yaml")
    config["evaluation"]["dag_quality"]["use_proxy_probe"] = True
    summary = build_dag_quality_summary(state, protocol, config, catalog)

    assert summary.proxy_probe_enabled is True
    assert summary.proxy_probe_macro_f1 is not None
    assert 0.0 <= summary.proxy_probe_macro_f1 <= 1.0
    assert summary.dataset_level["proxy_probe_enabled"] is True
    assert summary.dataset_level["proxy_probe_macro_f1"] is not None
    assert summary.recommendation_hint in {
        "finish_candidate",
        "patch_candidate",
        "replan_candidate",
        "halt_candidate",
    }
