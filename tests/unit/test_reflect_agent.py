from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent, reflect_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.evaluation import build_dag_quality_summary
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import ExecutionGap, WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _executed_state(config_name: str) -> tuple[WorkflowState, object]:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Reflect on the current DAG.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "POST_EXECUTE"},
    )
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    state.dag_quality_summary = build_dag_quality_summary(state, protocol, config, catalog).model_dump()
    return state, llm


def test_reflect_agent_returns_structured_finish_decision():
    state, llm = _executed_state("config/runs/rm101_synth_dag.yaml")
    state = reflect_agent(state, llm)

    result = state.reflection_results[-1]
    assert result.decision == "finish"
    assert isinstance(result.reason, str) and result.reason
    assert state.reflection_history[-1] == result.reason


def test_reflect_agent_requests_replan_when_execution_gaps_exist():
    state, llm = _executed_state("config/runs/rm101_synth_dag.yaml")
    state.execution_gaps.append(
        ExecutionGap(
            step_index=99,
            parent="ch1",
            op_name="decision.rule_based_decision",
            message="Unknown or unsupported operator: decision.rule_based_decision",
            recoverable=False,
        )
    )

    state = reflect_agent(state, llm)

    result = state.reflection_results[-1]
    assert result.decision == "need_replan"
    assert result.missing_operators


def test_reflect_agent_consumes_dag_quality_summary_for_patch_decision():
    state, llm = _executed_state("config/runs/rm101_synth_dag.yaml")
    state.execution_gaps.clear()
    state.dag_quality_summary = {
        "current_depth": 2,
        "min_depth": 2,
        "max_depth": 8,
        "depth_ok": True,
        "feature_node_count": 3,
        "multi_node_count": 1,
        "operator_categories": ["TRANSFORM", "AGGREGATE", "MULTI_VARIABLE"],
        "execution_gap_count": 0,
        "nan_ratio": 0.0,
        "zero_variance_ratio": 0.0,
        "proxy_probe_enabled": False,
        "proxy_probe_macro_f1": None,
        "dataset_level": {
            "enabled": True,
            "source": "split_subset_execution",
            "evidence_path": "ml",
            "materialization_ok": True,
            "all_finite": True,
            "distinguishable": True,
            "split_window_counts": {"train": 8, "val": 8, "test": 8},
            "feature_dims": {"train": 4, "val": 4, "test": 4},
            "decision_summary": {"node_count": 1},
            "issues": ["Proxy evidence is still weak."],
            "critical_failure": False,
        },
        "issues": ["Proxy evidence is still weak."],
        "recommendation_hint": "patch_candidate",
    }

    state = reflect_agent(state, llm)

    result = state.reflection_results[-1]
    assert result.decision == "need_patch"
    assert result.structural_warnings
