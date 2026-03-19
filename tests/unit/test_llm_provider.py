from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from src.agents import execute_agent, plan_agent, reflect_agent, report_agent
from src.bridge import compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.llm import CodexCliLLM, LLMProviderError, LLMSchemaError, OfflineLLM, OpenAICodexLLM, OpenRouterLLM, get_llm
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


def _mock_codex_exec(monkeypatch: pytest.MonkeyPatch, outputs: list[dict[str, object]], *, expected_model: str = "gpt-5.3-codex") -> None:
    monkeypatch.setattr("src.llm.client.shutil.which", lambda _: "/usr/bin/codex")

    def fake_run(cmd, input=None, text=None, capture_output=None, timeout=None, cwd=None):  # type: ignore[override]
        assert cmd[0] == "/usr/bin/codex"
        assert "exec" in cmd
        if "-c" in cmd:
            cfg_values = [cmd[index + 1] for index, token in enumerate(cmd[:-1]) if token == "-c"]
            assert any(value.startswith('model_reasoning_effort="') for value in cfg_values)
            assert any(value.startswith('plan_mode_reasoning_effort="') for value in cfg_values)
        if "-a" in cmd:
            assert cmd[cmd.index("-a") + 1] == "never"
        assert "-m" in cmd
        assert cmd[cmd.index("-m") + 1] == expected_model
        assert "-o" in cmd
        output_path = Path(cmd[cmd.index("-o") + 1])
        payload = outputs.pop(0)
        output_path.write_text(str(payload.get("output", "")), encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd,
            int(payload.get("returncode", 0)),
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
        )

    monkeypatch.setattr("src.llm.client.subprocess.run", fake_run)


def test_get_llm_dispatches_offline_codex_cli_openai_and_openrouter_providers():
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    llm = get_llm(config)
    assert isinstance(llm, OfflineLLM)

    config["llm"]["mode"] = "provider"
    llm = get_llm(config)
    assert isinstance(llm, CodexCliLLM)
    assert llm.api_key_env == ""

    config["llm"]["provider"] = "openai"
    llm = get_llm(config)
    assert isinstance(llm, OpenAICodexLLM)
    assert llm.api_key_env == "OPENAI_API_KEY"
    assert llm.base_url == "https://api.openai.com/v1"

    config["llm"]["provider"] = "openrouter"
    config["llm"]["model"] = "openai/gpt-4.1-mini"
    llm = get_llm(config)
    assert isinstance(llm, OpenRouterLLM)


def test_llm_public_imports_match_client_shim_exports():
    from src.llm import CodexCliLLM as public_codex
    from src.llm import OfflineLLM as public_offline
    from src.llm import get_llm as public_get_llm
    from src.llm.client import CodexCliLLM as shim_codex
    from src.llm.client import OfflineLLM as shim_offline
    from src.llm.client import get_llm as shim_get_llm

    assert public_get_llm is shim_get_llm
    assert public_codex is shim_codex
    assert public_offline is shim_offline


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


def test_openrouter_stepfun_text_mode_planner_accepts_dsl(monkeypatch: pytest.MonkeyPatch):
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
                            "content": '- parent=ch1 op=normalize params={"eps": 1e-6}\n- parent=ch2 op=normalize params={"eps": 1e-6}',
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
    assert [step.parent for step in plan.plan] == ["ch1", "ch2"]
    assert all(step.op_name == "normalize" for step in plan.plan)


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


def test_openrouter_stepfun_text_mode_reflector_accepts_structured_text(monkeypatch: pytest.MonkeyPatch):
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
                            "content": (
                                "Decision: patch\n"
                                "Reason: execution is acceptable but needs one more round.\n"
                                "Missing Operators:\n"
                                "- band_power\n"
                                "Shape Risks:\n"
                                "- none\n"
                                "Structural Warnings:\n"
                                "- continue expanding the spectral branch\n"
                            )
                        }
                    }
                ]
            },
        )

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler))
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
    assert result.decision == "need_patch"
    assert result.missing_operators == ["band_power"]
    assert result.structural_warnings


def test_openrouter_stepfun_reflector_repairs_plain_prose_once(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen_prompts: list[str] = []

    responses = [
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "The DAG looks acceptable. Please continue with another round."}}]},
        ),
        httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Decision: patch\nReason: continue with one more round.\nMissing Operators:\n- none\nShape Risks:\n- none\nStructural Warnings:\n- grow frequency diversity"}}]},
        ),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_prompts.append(payload["messages"][0]["content"])
        return responses.pop(0)

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler), retry_once=True)
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
    assert result.decision == "need_patch"
    assert len(seen_prompts) == 2
    assert "Normalize the following reflection response" in seen_prompts[1]


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


def test_openrouter_provider_retries_once_on_429(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {}}]})}}]},
        )

    llm = OpenRouterLLM(http_client=_mock_client(handler), retry_once=True)
    plan = llm.generate_step_plan(
        prompt="planner prompt",
        instruction="plan",
        signal_context=_signal_context(),
        dag_json=None,
        reflection=[],
        operator_catalog_summary=[],
    )
    assert plan.plan[0].op_name == "normalize"
    assert call_count == 2


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


def test_openai_codex_planner_and_reflector_use_strict_json_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen_response_formats: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_response_formats.append(payload["response_format"])
        prompt = payload["messages"][0]["content"]
        if prompt == "planner prompt":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {"eps": 1e-6}}]})}}]},
            )
        if prompt == "reflect prompt":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps({"decision": "finish", "reason": "ok", "missing_operators": [], "shape_risks": [], "structural_warnings": []})}}]},
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")

    llm = OpenAICodexLLM(http_client=_mock_client(handler))
    plan = llm.generate_step_plan(
        prompt="planner prompt",
        instruction="plan",
        signal_context=_signal_context(),
        dag_json=None,
        reflection=[],
        operator_catalog_summary=[],
    )
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
    assert plan.plan[0].op_name == "normalize"
    assert result.decision == "finish"
    assert seen_response_formats == [{"type": "json_object"}, {"type": "json_object"}]


def test_codex_cli_planner_and_reflector_use_output_schema(monkeypatch: pytest.MonkeyPatch):
    planner_schema_seen = False
    reflector_schema_seen = False

    monkeypatch.setattr("src.llm.client.shutil.which", lambda _: "/usr/bin/codex")

    def fake_run(cmd, input=None, text=None, capture_output=None, timeout=None, cwd=None):  # type: ignore[override]
        output_path = Path(cmd[cmd.index("-o") + 1])
        schema_path = Path(cmd[cmd.index("--output-schema") + 1])
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        nonlocal planner_schema_seen, reflector_schema_seen
        if "plan" in schema_payload.get("properties", {}):
            planner_schema_seen = True
            output_path.write_text(json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params_json": "{\"eps\": 1e-6}"}]}), encoding="utf-8")
        else:
            reflector_schema_seen = True
            output_path.write_text(json.dumps({"decision": "finish", "reason": "ok", "missing_operators": [], "shape_risks": [], "structural_warnings": []}), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("src.llm.client.subprocess.run", fake_run)

    llm = CodexCliLLM()
    plan = llm.generate_step_plan(
        prompt="planner prompt",
        instruction="plan",
        signal_context=_signal_context(),
        dag_json=None,
        reflection=[],
        operator_catalog_summary=[],
    )
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
    assert plan.plan[0].op_name == "normalize"
    assert result.decision == "finish"
    assert planner_schema_seen
    assert reflector_schema_seen


def test_codex_cli_param_resolution_uses_structured_schema(monkeypatch: pytest.MonkeyPatch):
    _mock_codex_exec(
        monkeypatch,
        outputs=[{"output": json.dumps({"custom_tau": 0.25})}],
    )
    llm = CodexCliLLM()
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


def test_codex_cli_provider_raises_on_subprocess_failure(monkeypatch: pytest.MonkeyPatch):
    _mock_codex_exec(
        monkeypatch,
        outputs=[{"returncode": 2, "stderr": "authentication required"}],
    )
    llm = CodexCliLLM()
    with pytest.raises(LLMProviderError):
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


def test_stepfun_provider_can_run_four_agents_end_to_end(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Build a PHM feature extraction DAG.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "FINAL_REPORT"},
    )

    responses = [
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '- parent=ch1 op=normalize params={"eps": 1e-6}\n- parent=ch2 op=normalize params={"eps": 1e-6}'
                        }
                    }
                ]
            },
        ),
        httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Decision: finish\n"
                                "Reason: the current DAG is valid for compilation.\n"
                                "Missing Operators:\n"
                                "- none\n"
                                "Shape Risks:\n"
                                "- none\n"
                                "Structural Warnings:\n"
                                "- none\n"
                            )
                        }
                    }
                ]
            },
        ),
        httpx.Response(200, json={"choices": [{"message": {"content": "# Provider Report\n\nok"}}]}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    llm = OpenRouterLLM(model="stepfun/step-3.5-flash:free", http_client=_mock_client(handler))
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    state = reflect_agent(state, llm)
    compiled = compile_dag_for_path(state.dag, "dag_only")
    report = report_agent(
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

    assert state.step_plan is not None
    assert state.reflection_results[-1].decision == "finish"
    assert report.startswith("# Provider Report")


def test_codex_cli_provider_can_run_four_agents_end_to_end(monkeypatch: pytest.MonkeyPatch):
    config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")
    protocol = build_protocol_from_config(config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Build a PHM feature extraction DAG.",
        dataset_name=protocol.dataset_name,
        graph_path=config["experiment"]["graph_path"],
        data_context={"min_depth": 2, "min_width": 1, "max_depth": 8, "stage": "FINAL_REPORT"},
    )

    _mock_codex_exec(
        monkeypatch,
        outputs=[
            {"output": json.dumps({"plan": [{"parent": "ch1", "op_name": "normalize", "params": {"eps": 1e-6}}]})},
            {"output": json.dumps({"decision": "finish", "reason": "ok", "missing_operators": [], "shape_risks": [], "structural_warnings": []})},
            {"output": "# Codex Report\n\nok"},
        ],
    )

    llm = CodexCliLLM()
    state = plan_agent(state, protocol, llm, catalog)
    state = execute_agent(state, protocol, catalog, llm)
    state = reflect_agent(state, llm)
    compiled = compile_dag_for_path(state.dag, "dag_only")
    report = report_agent(
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

    assert state.step_plan is not None
    assert state.reflection_results[-1].decision == "finish"
    assert report.startswith("# Codex Report")
