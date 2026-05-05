from __future__ import annotations

from pathlib import Path

from src.agents import execute_agent, plan_agent
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.llm import LLMProviderError, get_llm
from src.operators import get_operator_catalog
from src.prompts.plan_prompt import PLAN_PROMPT_INPUT_FIELDS, PLAN_PROMPT_OUTPUT_FIELDS, render_plan_prompt
from src.states import WorkflowState


ROOT = Path(__file__).resolve().parents[2]


class _RateLimitedPlanLLM:
    provider = "test"
    mode = "provider"
    model = "rate-limited-planner"
    api_key_env = "TEST_API_KEY"

    def generate_step_plan(self, **kwargs):
        raise LLMProviderError("rate limited")


class _TransformOnlyPlanLLM:
    provider = "test"
    mode = "provider"
    model = "transform-only-planner"
    api_key_env = "TEST_API_KEY"

    def generate_step_plan(self, **kwargs):
        from src.states import StepPlan

        return StepPlan.model_validate(
            {
                "plan": [
                    {"parent": "ch1", "op_name": "normalize", "params": {}},
                    {"parent": "ch1", "op_name": "fft", "params": {}},
                ]
            }
        )


def test_plan_agent_outputs_nvta_style_step_plan():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Build a PHM feature extraction DAG.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8},
    )

    state = plan_agent(state, protocol, llm, catalog)

    assert state.signal_context is not None
    assert state.signal_context.root_node_ids == [f"ch{i + 1}" for i in range(state.signal_context.channel_count)]
    assert state.step_plan is not None
    payload = state.step_plan.model_dump()
    assert list(payload) == ["plan"]
    assert payload["plan"]
    assert all(set(step.keys()) == {"parent", "op_name", "params"} for step in payload["plan"])
    assert payload["plan"][0]["parent"].startswith("ch")
    planned_ops = {step["op_name"] for step in payload["plan"]}
    assert {"stft", "patch", "cross_correlation", "threshold"} <= planned_ops


def test_plan_prompt_contract_matches_agent_inputs_and_outputs():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    state = WorkflowState(
        user_instruction="Plan the next DAG layer.",
        dataset_name=protocol.dataset_name,
        graph_path="dag_only",
    )
    llm = get_llm(config)
    catalog = get_operator_catalog()
    state = plan_agent(state, protocol, llm, catalog)
    prompt = render_plan_prompt(
        instruction=state.user_instruction,
        signal_context=state.signal_context.model_dump(),
        dag_json=None,
        tools=catalog.summary(),
        reflection=[],
        current_depth=0,
        min_depth=2,
        min_width=1,
    )
    assert "instruction" in PLAN_PROMPT_INPUT_FIELDS
    assert any("plan" in field for field in PLAN_PROMPT_OUTPUT_FIELDS)
    assert "Signal context" in prompt
    assert "Current DAG" in prompt


def test_plan_agent_falls_back_to_deterministic_feature_plan_on_provider_error():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Build a PHM feature extraction DAG despite provider rate limits.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 3, "min_width": 1, "max_depth": 8},
    )

    state = plan_agent(state, protocol, _RateLimitedPlanLLM(), catalog)

    assert state.step_plan is not None
    planned_ops = [step.op_name for step in state.step_plan.plan]
    assert "normalize" in planned_ops
    assert "fft" in planned_ops
    assert {"rms", "mean", "std"} <= set(planned_ops)
    assert any("provider_plan_fallback" in item for item in state.reflection_history)

    state = execute_agent(state, protocol, catalog, _RateLimitedPlanLLM())

    assert not state.execution_gaps
    assert state.dag is not None
    assert any(node.kind == "feature" for node in state.dag.nodes)


def test_plan_agent_replaces_transform_only_provider_plan_for_model_paths():
    config = load_runtime_config(ROOT / "config/runs/ottawa_synth_ml.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Build an ML DAG even if the provider omits feature outputs.",
        dataset_name=protocol.dataset_name,
        graph_path="ml",
        data_context={"min_depth": 3, "min_width": 1, "max_depth": 8},
    )

    state = plan_agent(state, protocol, _TransformOnlyPlanLLM(), catalog)

    assert state.step_plan is not None
    planned_ops = {step.op_name for step in state.step_plan.plan}
    assert {"mean", "std", "rms"} <= planned_ops
    assert any("deterministic_feature_plan_guard" in item for item in state.reflection_history)

    state = execute_agent(state, protocol, catalog, _TransformOnlyPlanLLM())

    assert not state.execution_gaps
    assert state.dag is not None
    assert any(node.kind == "feature" for node in state.dag.nodes)
