from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from src.agents import execute_agent, plan_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_preview_signal
from src.llm import LLMClient, get_llm
from src.operators import get_operator_catalog
from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan, WorkflowState


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
    assert any(node.kind == "decision" for node in state.dag.nodes)
    assert any(node.operator_category == "EXPAND" for node in state.dag.nodes)
    assert any(node.rank_class == "terminal_decision" for node in state.dag.nodes if node.kind == "decision")
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


class SpyLLM:
    provider = "test"
    mode = "provider"
    model = "test-model"
    api_key_env = ""

    def __init__(self, params_to_return: Optional[Dict[str, Any]] = None) -> None:
        self.params_to_return = params_to_return or {}
        self.param_calls = 0

    def generate_step_plan(
        self,
        *,
        prompt: str,
        instruction: str,
        signal_context: SignalContext,
        dag_json: Optional[Dict[str, Any]],
        reflection: Iterable[str],
        operator_catalog_summary: Iterable[Dict[str, Any]],
        trace_context: Optional[Dict[str, Any]] = None,
    ) -> StepPlan:
        raise NotImplementedError

    def resolve_missing_params(
        self,
        *,
        prompt: str,
        op_name: str,
        param_schema: Dict[str, str],
        param_defaults: Dict[str, Any],
        param_docs: Dict[str, str],
        llm_tunable_params: List[str],
        provided_params: Dict[str, Any],
        signal_context: SignalContext,
        parent_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.param_calls += 1
        return dict(self.params_to_return)

    def reflect_workflow(
        self,
        *,
        prompt: str,
        instruction: str,
        stage: str,
        dag_blueprint: Dict[str, Any],
        dag_quality_summary: Dict[str, Any],
        issues_summary: str,
        min_depth: int,
        min_width: int,
        max_depth: int,
        current_depth: int,
        execution_gaps: List[ExecutionGap],
    ) -> ReflectionResult:
        raise NotImplementedError

    def render_report(
        self,
        *,
        prompt: str,
        instruction: str,
        dataset_name: str,
        graph_path: str,
        compiled_manifest: Dict[str, Any],
        path_artifacts: Dict[str, Any],
        reflection_summary: Dict[str, Any],
        dag_quality_summary: Dict[str, Any],
        review_context: Dict[str, Any],
        step_plan: Dict[str, Any],
    ) -> str:
        raise NotImplementedError


def _manual_state(config_name: str, step_plan: Dict[str, Any]) -> tuple[WorkflowState, object, object]:
    config = load_runtime_config(ROOT / config_name)
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    sample_id, preview_window = materialize_preview_signal(protocol)
    state = WorkflowState(
        user_instruction="Execute the planned PHM pipeline.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        runtime_config=config,
        signal_context=SignalContext(
            dataset_name=protocol.dataset_name,
            channel_count=int(preview_window.shape[0]),
            window_shape=list(preview_window.shape),
            sampling_rate=int(protocol.samples[0].sampling_rate),
            source_mode=protocol.source_mode,
            root_node_ids=[f"ch{i + 1}" for i in range(int(preview_window.shape[0]))],
            representative_sample_id=sample_id,
        ),
        step_plan=StepPlan.model_validate(step_plan),
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8},
    )
    return state, protocol, catalog


def test_execute_agent_short_circuits_param_resolution_when_no_tunable_is_missing():
    state, protocol, catalog = _manual_state(
        "config/runs/rm101_synth_ml.yaml",
        {"plan": [{"parent": "ch1", "op_name": "normalize", "params": {"eps": 1e-6}}]},
    )
    spy = SpyLLM()

    state = execute_agent(state, protocol, catalog, spy)

    assert spy.param_calls == 0
    assert any(node.op_uid == "signal.normalize" for node in state.dag.nodes)


class FakeOperator:
    def __init__(self) -> None:
        self.spec = SimpleNamespace(
            op_uid="signal.fake_tunable",
            op_name="fake_tunable",
            name="Fake Tunable",
            schema_category="TRANSFORM",
            rank_class="rank_same",
            input_spec={"arity": "single", "min_rank": 2},
            output_spec={"semantic": "channel_first_signal", "rank_behavior": "preserve"},
            param_schema={"custom_tau": "float"},
            param_defaults={},
            param_docs={"custom_tau": "Synthetic tunable parameter used only for unit coverage."},
            backend_availability=["np"],
            execution_role="fixed",
            legal_paths=["dag_only", "ml", "torch"],
            llm_tunable_params=["custom_tau"],
        )

    def forward_np(self, x: np.ndarray, **kwargs: float) -> np.ndarray:
        return np.asarray(x, dtype=float) * float(kwargs["custom_tau"])


class FakeCatalog:
    def __init__(self) -> None:
        self.operator = FakeOperator()

    def get_by_plan_name(self, op_name: str) -> FakeOperator:
        if op_name != "fake_tunable":
            raise KeyError(op_name)
        return self.operator


def test_execute_agent_calls_param_resolution_only_for_missing_tunable_params():
    state, protocol, _ = _manual_state(
        "config/runs/rm101_synth_ml.yaml",
        {"plan": [{"parent": "ch1", "op_name": "fake_tunable", "params": {}}]},
    )
    catalog = FakeCatalog()
    spy = SpyLLM(params_to_return={"custom_tau": 0.25})

    state = execute_agent(state, protocol, catalog, spy)

    assert spy.param_calls == 1
    assert any(node.op_uid == "signal.fake_tunable" for node in state.dag.nodes)


def test_execute_agent_supervisor_proving_rejects_tunable_ops_without_param_resolution():
    state, protocol, _ = _manual_state(
        "config/runs/rm101_synth_ml.yaml",
        {"plan": [{"parent": "ch1", "op_name": "fake_tunable", "params": {}}]},
    )
    state.runtime_config = {
        "runtime": {"workflow_mode": "supervisor_proving"},
        "experiment": {"graph_path": "ml"},
    }
    catalog = FakeCatalog()
    spy = SpyLLM(params_to_return={"custom_tau": 0.25})

    state = execute_agent(state, protocol, catalog, spy)

    assert spy.param_calls == 0
    assert state.execution_gaps
    assert "does not allow LLM-tunable params" in state.execution_gaps[0].message
