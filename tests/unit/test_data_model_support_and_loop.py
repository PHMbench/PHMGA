from __future__ import annotations

from pathlib import Path

from scripts.run_case import _run_frontend_loop
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_dataset_views, build_protocol_from_config, materialize_split_signals
from src.llm.client import OfflineLLM
from src.model import build_similarity_artifacts, run_shallow_ml_baseline
from src.operators import get_operator_catalog
from src.states import StepPlan, WorkflowState


ROOT = Path(__file__).resolve().parents[2]


class ReplanOnceLLM(OfflineLLM):
    def __init__(self) -> None:
        super().__init__()
        self._plan_calls = 0

    def generate_step_plan(self, **kwargs):  # type: ignore[override]
        self._plan_calls += 1
        if self._plan_calls == 1:
            return StepPlan.model_validate({"plan": [{"parent": "ch1", "op_name": "unknown_op", "params": {}}]})
        return StepPlan.model_validate({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {"eps": 1e-6}}]})


def _ml_state_and_artifacts():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    llm = OfflineLLM()
    state = WorkflowState(
        user_instruction="Build a shallow PHM baseline.",
        dataset_name=protocol.dataset_name,
        graph_path="ml",
        max_iterations=4,
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "TEST"},
    )
    state = _run_frontend_loop(state, protocol, llm, catalog)
    compiled = compile_dag_for_path(state.dag, "ml")
    split_records = materialize_split_signals(protocol)
    return protocol, catalog, compiled, split_records


def test_dataset_preparer_and_model_side_support_modules_work_together():
    protocol, catalog, compiled, split_records = _ml_state_and_artifacts()
    dataset_views = build_dataset_views(compiled, split_records, catalog)

    assert set(dataset_views) == {"train", "val", "test"}
    assert dataset_views["train"].X.ndim == 2
    assert dataset_views["train"].X.shape[0] == len(dataset_views["train"].sample_ids)

    baseline = run_shallow_ml_baseline(dataset_views, "logistic_regression", max_iter=100)
    assert "metrics" in baseline and "test" in baseline["metrics"]
    assert "predictions" in baseline and baseline["predictions"]["test"]

    similarity = build_similarity_artifacts(dataset_views)
    assert "split_sizes" in similarity
    assert similarity["split_sizes"]["train"] > 0
    assert "class_centroid_similarity" in similarity
    assert protocol.dataset_name == "RM101_SYNTH"


def test_frontend_loop_rolls_back_on_need_replan():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    llm = ReplanOnceLLM()
    state = WorkflowState(
        user_instruction="Grow a valid DAG after one failed round.",
        dataset_name=protocol.dataset_name,
        graph_path="dag_only",
        max_iterations=3,
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "TEST"},
    )

    state = _run_frontend_loop(state, protocol, llm, catalog)

    assert len(state.round_history) >= 2
    assert state.round_history[0].rolled_back is True
    assert state.round_history[0].reflection_result is not None
    assert state.round_history[0].reflection_result.decision == "need_replan"
    assert state.dag is not None
    assert any(node.op_uid == "signal.normalize" for node in state.dag.nodes)
