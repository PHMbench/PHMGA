from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.agents import execute_agent, plan_agent, reflect_agent, report_agent
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.llm import LLMProviderError, LLMSchemaError, OfflineLLM, OpenRouterLLM, get_llm
from src.operators import get_operator_catalog
from src.states import ReflectionResult, SignalContext, StepPlan, WorkflowState


ROOT = Path(__file__).resolve().parents[2]


def _signal_context() -> SignalContext:
    return SignalContext(
        dataset_name="RM101_SYNTH",
        channel_count=2,
        window_shape=[2, 128],
        sampling_rate=25600,
        source_mode="synthetic",
        root_node_ids=["ch1", "ch2"],
    )


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_get_llm_dispatches_offline_and_openrouter_provider():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    llm = get_llm(config)
    assert isinstance(llm, OfflineLLM)

    config["llm"]["mode"] = "provider"
    config["llm"]["model"] = "openai/gpt-4.1-mini"
    llm = get_llm(config)
    assert isinstance(llm, OpenRouterLLM)


def test_openrouter_planner_parses_structured_step_plan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["response_format"]["type"] == "json_object"
        assert payload["messages"][0]["content"] == "planner prompt"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {}}]})}}]},
        )

    llm = OpenRouterLLM(http_client=_mock_client(handler))
    plan = llm.generate_step_plan(
        prompt="planner prompt",
        instruction="plan",
        signal_context=_signal_context(),
        dag_json=None,
        reflection=[],
        operator_catalog_summary=[],
    )
    assert isinstance(plan, StepPlan)
    assert plan.plan[0].op_name == "normalize"


def test_openrouter_stepfun_text_mode_planner_uses_reasoning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert "response_format" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {}}]}),
                        }
                    }
                ]
            },
        )

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler))
    plan = llm.generate_step_plan(
        prompt="planner prompt",
        instruction="plan",
        signal_context=_signal_context(),
        dag_json=None,
        reflection=[],
        operator_catalog_summary=[],
    )
    assert isinstance(plan, StepPlan)
    assert plan.plan[0].op_name == "normalize"


def test_openrouter_param_resolution_accepts_only_requested_tunable_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["messages"][0]["content"] == "param prompt"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"custom_tau": 0.25})}}]},
        )

    llm = OpenRouterLLM(http_client=_mock_client(handler))
    params = llm.resolve_missing_params(
        prompt="param prompt",
        op_name="custom.wavefilter",
        param_schema={"custom_tau": "float"},
        param_defaults={},
        param_docs={"custom_tau": "Temperature-like scalar."},
        llm_tunable_params=["custom_tau"],
        provided_params={},
        signal_context=_signal_context(),
        parent_summaries=[{"node_id": "ch1", "shape": [1, 128]}],
    )
    assert params["custom_tau"] == 0.25


def test_openrouter_text_mode_param_resolution_parses_fenced_json(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert "response_format" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "Use this value:\n```json\n{\"custom_tau\": 0.25}\n```",
                            "reasoning": None,
                        }
                    }
                ]
            },
        )

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler))
    params = llm.resolve_missing_params(
        prompt="param prompt",
        op_name="custom.wavefilter",
        param_schema={"custom_tau": "float"},
        param_defaults={},
        param_docs={"custom_tau": "Temperature-like scalar."},
        llm_tunable_params=["custom_tau"],
        provided_params={},
        signal_context=_signal_context(),
        parent_summaries=[{"node_id": "ch1", "shape": [1, 128]}],
    )
    assert params["custom_tau"] == 0.25


def test_openrouter_param_resolution_rejects_illegal_keys(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"custom_tau": 0.25, "illegal": 1})}}]},
        )

    llm = OpenRouterLLM(http_client=_mock_client(handler))
    with pytest.raises(LLMSchemaError):
        llm.resolve_missing_params(
            prompt="param prompt",
            op_name="custom.wavefilter",
            param_schema={"custom_tau": "float"},
            param_defaults={},
            param_docs={"custom_tau": "Temperature-like scalar."},
            llm_tunable_params=["custom_tau"],
            provided_params={},
            signal_context=_signal_context(),
            parent_summaries=[{"node_id": "ch1", "shape": [1, 128]}],
        )


def test_openrouter_text_mode_reflector_rejects_non_json_reasoning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert "response_format" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning": "plain prose without json",
                        }
                    }
                ]
            },
        )

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler))
    with pytest.raises(LLMSchemaError):
        llm.reflect_workflow(
            prompt="reflect prompt",
            instruction="reflect",
            stage="POST_EXECUTE",
            dag_blueprint={"nodes": [{"node_id": "ch1"}], "edges": []},
            dag_quality_summary={"issues": []},
            issues_summary="",
            min_depth=2,
            min_width=1,
            max_depth=8,
            current_depth=2,
            execution_gaps=[],
        )


def test_openrouter_reflector_parses_structured_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content.decode("utf-8"))["messages"][0]["content"] == "reflect prompt"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                ReflectionResult(
                                    decision="finish",
                                    reason="Looks good.",
                                    missing_operators=[],
                                    shape_risks=[],
                                    structural_warnings=[],
                                ).model_dump()
                            )
                        }
                    }
                ]
            },
        )

    llm = OpenRouterLLM(http_client=_mock_client(handler))
    result = llm.reflect_workflow(
        prompt="reflect prompt",
        instruction="reflect",
        stage="POST_EXECUTE",
        dag_blueprint={"nodes": [{"node_id": "ch1"}], "edges": []},
        dag_quality_summary={"issues": []},
        issues_summary="",
        min_depth=2,
        min_width=1,
        max_depth=8,
        current_depth=2,
        execution_gaps=[],
    )
    assert result.decision == "finish"


def test_openrouter_reporter_returns_markdown(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert "report prompt" in payload["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "# Provider Report\n\nok"}}]})

    llm = OpenRouterLLM(http_client=_mock_client(handler))
    report = llm.render_report(
        prompt="report prompt",
        instruction="report",
        dataset_name="RM101_SYNTH",
        graph_path="ml",
        compiled_manifest={"dag_hash": "abc"},
        path_artifacts={},
        reflection_summary={},
        dag_quality_summary={},
        review_context={},
        step_plan={"plan": []},
    )
    assert report.startswith("# Provider Report")


def test_openrouter_provider_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm = OpenRouterLLM(http_client=_mock_client(lambda request: httpx.Response(200, json={})))
    with pytest.raises(LLMProviderError):
        llm.generate_step_plan(
            prompt="planner prompt",
            instruction="plan",
            signal_context=_signal_context(),
            dag_json=None,
            reflection=[],
            operator_catalog_summary=[],
        )


def test_openrouter_provider_raises_on_http_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = OpenRouterLLM(http_client=_mock_client(lambda request: httpx.Response(401, json={"error": "unauthorized"})))
    with pytest.raises(LLMProviderError):
        llm.generate_step_plan(
            prompt="planner prompt",
            instruction="plan",
            signal_context=_signal_context(),
            dag_json=None,
            reflection=[],
            operator_catalog_summary=[],
        )


def test_openrouter_provider_rejects_invalid_json_payload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = OpenRouterLLM(
        http_client=_mock_client(
            lambda request: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]})
        )
    )
    with pytest.raises(LLMSchemaError):
        llm.generate_step_plan(
            prompt="planner prompt",
            instruction="plan",
            signal_context=_signal_context(),
            dag_json=None,
            reflection=[],
            operator_catalog_summary=[],
        )


class PromptCaptureLLM(OfflineLLM):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: dict[str, str] = {}

    def generate_step_plan(self, **kwargs):  # type: ignore[override]
        self.prompts["plan"] = kwargs["prompt"]
        return super().generate_step_plan(**kwargs)

    def resolve_missing_params(self, **kwargs):  # type: ignore[override]
        self.prompts["param"] = kwargs["prompt"]
        return super().resolve_missing_params(**kwargs)

    def reflect_workflow(self, **kwargs):  # type: ignore[override]
        self.prompts["reflect"] = kwargs["prompt"]
        return super().reflect_workflow(**kwargs)

    def render_report(self, **kwargs):  # type: ignore[override]
        self.prompts["report"] = kwargs["prompt"]
        return super().render_report(**kwargs)


def test_agents_pass_rendered_prompts_to_llm():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    llm = PromptCaptureLLM()
    state = WorkflowState(
        user_instruction="Exercise prompt plumbing.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "FINAL_REPORT"},
    )

    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    state = reflect_agent(state, llm)
    compiled = compile_dag_for_path(state.dag, "dag_only")
    report_agent(
        state,
        protocol,
        compiled.manifest,
        {
            "node_inventory": compiled.node_inventory,
            "edge_inventory": compiled.edge_inventory,
            "method_description": compiled.method_description,
        },
        llm,
    )

    assert "Role: Planner" in llm.prompts["plan"]
    assert "Role: Parameter Resolver" in llm.prompts["param"]
    assert "Role: Reflector" in llm.prompts["reflect"]
    assert "Role: Reporter" in llm.prompts["report"]
