from __future__ import annotations

from src.agents.reflect_agent import reflect_agent
from src.builder_quality import (
    PHASE_COMBINE,
    PHASE_FEATURE,
    PHASE_RAW,
    evaluate_builder_richness,
    infer_builder_phase,
    validate_plan_steps,
)
from langgraph.graph import END

from src.phm_outer_graph import _outer_transition
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def _input_node(node_id: str) -> InputData:
    return InputData(
        node_id=node_id,
        parents=[],
        shape=(2, 32, 1),
        data={},
        results={"ref": {}, "tst": {}},
        metadata={},
        meta={"channel": node_id},
    )


def _processed_node(node_id: str, *, parent: str | list[str], method: str) -> ProcessedData:
    parent_ids = parent if isinstance(parent, list) else [parent]
    return ProcessedData(
        node_id=node_id,
        parents=parent_ids,
        shape=(2, 32, 1),
        source_signal_id=",".join(parent_ids),
        method=method,
        results={"ref": {}, "tst": {}},
        meta={"tool": method, "parent": ",".join(parent_ids)},
    )


def _state_with_nodes(nodes: dict[str, object], leaves: list[str], *, min_depth: int = 3, min_width: int = 2, max_depth: int = 4) -> PHMState:
    ch1 = nodes["ch1"]
    ch2 = nodes["ch2"]
    dag = DAGState(
        user_instruction="diagnose RM101",
        channels=["ch1", "ch2"],
        nodes=nodes,
        leaves=leaves,
    )
    return PHMState(
        case_name="case_exp2",
        user_instruction="diagnose RM101",
        reference_signal=ch1,
        test_signal=ch2,
        dag_state=dag,
        min_depth=min_depth,
        min_width=min_width,
        max_depth=max_depth,
        runtime_config={"llm": {"provider": "openrouter", "model": "z-ai/glm-4.5-air:free"}},
    )


def _raw_state() -> PHMState:
    ch1 = _input_node("ch1")
    ch2 = _input_node("ch2")
    return _state_with_nodes({"ch1": ch1, "ch2": ch2}, ["ch1", "ch2"])


def _transform_state() -> PHMState:
    ch1 = _input_node("ch1")
    ch2 = _input_node("ch2")
    fft_01 = _processed_node("fft_01_ch1", parent="ch1", method="fft")
    stft_01 = _processed_node("stft_01_ch2", parent="ch2", method="stft")
    return _state_with_nodes(
        {"ch1": ch1, "ch2": ch2, "fft_01_ch1": fft_01, "stft_01_ch2": stft_01},
        ["fft_01_ch1", "stft_01_ch2"],
    )


def _feature_state(*, min_depth: int = 4) -> PHMState:
    ch1 = _input_node("ch1")
    ch2 = _input_node("ch2")
    fft_01 = _processed_node("fft_01_ch1", parent="ch1", method="fft")
    stft_01 = _processed_node("stft_01_ch2", parent="ch2", method="stft")
    band_01 = _processed_node("band_power_01_ch1", parent="fft_01_ch1", method="band_power")
    kurt_01 = _processed_node("kurtosis_01_ch2", parent="stft_01_ch2", method="kurtosis")
    return _state_with_nodes(
        {
            "ch1": ch1,
            "ch2": ch2,
            "fft_01_ch1": fft_01,
            "stft_01_ch2": stft_01,
            "band_power_01_ch1": band_01,
            "kurtosis_01_ch2": kurt_01,
        },
        ["band_power_01_ch1", "kurtosis_01_ch2"],
        min_depth=min_depth,
    )


def _normalize_feature_state(*, min_depth: int = 3) -> PHMState:
    ch1 = _input_node("ch1")
    ch2 = _input_node("ch2")
    normalize_01 = _processed_node("normalize_01_ch1", parent="ch1", method="normalize")
    normalize_02 = _processed_node("normalize_02_ch2", parent="ch2", method="normalize")
    band_01 = _processed_node("band_power_01_ch1", parent="normalize_01_ch1", method="band_power")
    kurt_01 = _processed_node("kurtosis_01_ch2", parent="normalize_02_ch2", method="kurtosis")
    return _state_with_nodes(
        {
            "ch1": ch1,
            "ch2": ch2,
            "normalize_01_ch1": normalize_01,
            "normalize_02_ch2": normalize_02,
            "band_power_01_ch1": band_01,
            "kurtosis_01_ch2": kurt_01,
        },
        ["band_power_01_ch1", "kurtosis_01_ch2"],
        min_depth=min_depth,
        max_depth=3,
    )


def test_infer_builder_phase_distinguishes_raw_feature_and_combine():
    assert infer_builder_phase(_raw_state()) == PHASE_RAW
    assert infer_builder_phase(_transform_state()) == PHASE_FEATURE
    assert infer_builder_phase(_feature_state()) == PHASE_COMBINE


def test_validate_plan_steps_rejects_root_only_and_mean_only():
    state = _raw_state()
    ok, reason = validate_plan_steps(state, [])
    assert not ok
    assert "empty" in reason

    ok, reason = validate_plan_steps(
        state,
        [
            {"parent": "ch1", "op_name": "mean", "params": {}},
            {"parent": "ch2", "op_name": "mean", "params": {}},
        ],
        phase=PHASE_RAW,
    )
    assert not ok
    assert "mean-only" in reason or "weak reducer" in reason


def test_evaluate_builder_richness_rejects_trivial_dags_and_accepts_non_trivial_dag():
    raw_quality = evaluate_builder_richness(_raw_state())
    assert not raw_quality["passes"]
    assert raw_quality["is_root_only"]

    ch1 = _input_node("ch1")
    ch2 = _input_node("ch2")
    mean_01 = _processed_node("mean_01_ch1", parent="ch1", method="mean")
    mean_02 = _processed_node("mean_02_ch2", parent="ch2", method="mean")
    mean_state = _state_with_nodes({"ch1": ch1, "ch2": ch2, "mean_01_ch1": mean_01, "mean_02_ch2": mean_02}, ["mean_01_ch1", "mean_02_ch2"])
    mean_quality = evaluate_builder_richness(mean_state)
    assert not mean_quality["passes"]
    assert mean_quality["is_mean_only"]

    normalize_quality = evaluate_builder_richness(_normalize_feature_state())
    assert not normalize_quality["passes"]
    assert not normalize_quality["has_complete_strong_path"]

    rich_quality = evaluate_builder_richness(_feature_state(min_depth=3))
    assert rich_quality["passes"]
    assert rich_quality["has_spectral_transform"]
    assert rich_quality["has_feature_stat"]
    assert rich_quality["has_complete_strong_path"]


def test_validate_plan_steps_rejects_feature_ops_on_raw_inputs_and_accepts_spectral_then_feature():
    raw_state = _raw_state()
    ok, reason = validate_plan_steps(
        raw_state,
        [
            {"parent": "ch1", "op_name": "band_power", "params": {}},
            {"parent": "ch2", "op_name": "kurtosis", "params": {}},
        ],
        phase=PHASE_RAW,
    )
    assert not ok
    assert "spectral/time-frequency transform" in reason or "must not jump directly" in reason

    spectral_state = _transform_state()
    ok, reason = validate_plan_steps(
        spectral_state,
        [
            {"parent": "fft_01_ch1", "op_name": "band_power", "params": {}},
            {"parent": "stft_01_ch2", "op_name": "kurtosis", "params": {}},
        ],
        phase=PHASE_FEATURE,
    )
    assert ok, reason


def test_reflect_agent_provider_failure_requests_patch_when_quality_is_insufficient(monkeypatch):
    state = _transform_state()

    class FailingLLM:
        mode = "provider"

        def generate_json(self, prompt, repair_prompt=None):
            raise RuntimeError("429 rate limit")

    monkeypatch.setattr("src.agents.reflect_agent.get_llm", lambda runtime_config: FailingLLM())
    result = reflect_agent(
        instruction=state.user_instruction,
        stage="POST_EXECUTE",
        dag_blueprint={"nodes": []},
        issues_summary=None,
        state=state,
    )
    assert result["decision"] == "need_patch"
    assert "missing feature/stat op" in result["reason"] or "429" in result["reason"]


def test_reflect_agent_halts_at_max_depth_when_quality_is_still_insufficient(monkeypatch):
    state = _transform_state()
    state.max_depth = 2

    class FailingLLM:
        mode = "provider"

        def generate_json(self, prompt, repair_prompt=None):
            raise RuntimeError("timeout")

    monkeypatch.setattr("src.agents.reflect_agent.get_llm", lambda runtime_config: FailingLLM())
    result = reflect_agent(
        instruction=state.user_instruction,
        stage="POST_EXECUTE",
        dag_blueprint={"nodes": []},
        issues_summary=None,
        state=state,
    )
    assert result["decision"] == "halt"


def test_outer_transition_stops_on_halt():
    state = _feature_state()
    state.last_reflection_decision = "halt"
    assert _outer_transition(state) == END
