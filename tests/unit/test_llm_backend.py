from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from src.configuration import Configuration
from src.llm.base import LLMBackendError
from src.llm.backends import BigModelBackend, OpenRouterBackend, get_llm


def test_openrouter_backend_parses_json_successfully(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    backend = OpenRouterBackend()
    monkeypatch.setattr(
        backend,
        "_request",
        lambda prompt, expect_json: '{"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]}',
    )

    payload = backend.generate_json("planner prompt")

    assert payload["plan"][0]["op_name"] == "fft"


def test_openrouter_backend_uses_repair_lane_when_first_response_is_invalid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    backend = OpenRouterBackend()
    responses = iter(
        [
            "Here is your explanation instead of JSON.",
            '{"decision": "finish", "reason": "ok"}',
        ]
    )
    monkeypatch.setattr(backend, "_request", lambda prompt, expect_json: next(responses))

    payload = backend.generate_json("reflect prompt", repair_prompt="repair prompt")

    assert payload["decision"] == "finish"
    assert payload["reason"] == "ok"


def test_openrouter_backend_client_ignores_env_proxy_and_uses_split_timeouts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")

    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            captured["url"] = url
            captured["request_json"] = kwargs["json"]
            return DummyResponse()

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)

    backend = OpenRouterBackend(timeout_sec=42.0)
    payload = backend.generate_json("planner prompt")

    timeout = captured["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert captured["trust_env"] is False
    assert timeout.connect == 10.0
    assert timeout.write == 10.0
    assert timeout.pool == 5.0
    assert timeout.read == 42.0
    assert payload["ok"] is True


def test_openrouter_backend_wraps_connect_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            raise httpx.ConnectTimeout("connect timeout")

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)

    backend = OpenRouterBackend(timeout_sec=42.0)

    with pytest.raises(LLMBackendError, match=r"connect timeout after 10\.0s"):
        backend.generate_text("text prompt")


def test_openrouter_backend_wraps_read_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            raise httpx.ReadTimeout("read timeout")

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)

    backend = OpenRouterBackend(timeout_sec=17.0)

    with pytest.raises(LLMBackendError, match=r"read timeout after 17\.0s"):
        backend.generate_text("text prompt")


def test_openrouter_backend_retries_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    call_state = {"count": 0}

    class DummyResponse:
        def __init__(self, status_code: int, payload: dict[str, object], headers: dict[str, str] | None = None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("rate limited", request=self.request, response=self)

        def json(self) -> dict[str, object]:
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            call_state["count"] += 1
            if call_state["count"] == 1:
                return DummyResponse(429, {}, {"Retry-After": "0"})
            return DummyResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)
    monkeypatch.setattr("src.llm.backends.time.sleep", lambda seconds: None)

    backend = OpenRouterBackend(max_retries=3)
    payload = backend.generate_json("planner prompt")

    assert call_state["count"] == 2
    assert payload["ok"] is True


def test_agent_doc_mentions_proxy_cleanup_and_diagnostics():
    agent_doc = Path(__file__).resolve().parents[2] / "agent.md"
    text = agent_doc.read_text(encoding="utf-8")

    assert "env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY" in text
    assert "OPENROUTER_API_KEY" in text
    assert "/proc/<PID>/environ" in text
    assert ":443" in text


def test_bigmodel_backend_parses_json_successfully(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "dummy")
    backend = BigModelBackend()
    monkeypatch.setattr(
        backend,
        "_request",
        lambda prompt, expect_json: '{"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}]}',
    )

    payload = backend.generate_json("planner prompt")

    assert payload["plan"][0]["op_name"] == "fft"


def test_bigmodel_backend_uses_repair_lane_when_first_response_is_invalid(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "dummy")
    backend = BigModelBackend()
    responses = iter(
        [
            "Not JSON",
            '{"decision": "finish", "reason": "ok"}',
        ]
    )
    monkeypatch.setattr(backend, "_request", lambda prompt, expect_json: next(responses))

    payload = backend.generate_json("reflect prompt", repair_prompt="repair prompt")

    assert payload["decision"] == "finish"
    assert payload["reason"] == "ok"


def test_bigmodel_backend_disables_thinking_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "dummy")
    captured: dict[str, object] = {}

    class DummyResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            captured["json"] = kwargs["json"]
            return DummyResponse()

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)

    backend = BigModelBackend()
    payload = backend.generate_json("planner prompt")

    assert payload["ok"] is True
    assert captured["json"]["thinking"]["type"] == "disabled"


def test_bigmodel_backend_retries_429_with_retry_after(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "dummy")
    call_state = {"count": 0}

    class DummyResponse:
        def __init__(self, status_code: int, payload: dict[str, object], headers: dict[str, str] | None = None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}
            self.request = httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("rate limited", request=self.request, response=self)

        def json(self) -> dict[str, object]:
            return self._payload

    class DummyClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def post(self, url: str, **kwargs):
            call_state["count"] += 1
            if call_state["count"] == 1:
                return DummyResponse(429, {}, {"Retry-After": "0"})
            return DummyResponse(200, {"choices": [{"message": {"content": '{"ok": true}'}}]})

    monkeypatch.setattr("src.llm.backends.httpx.Client", DummyClient)
    monkeypatch.setattr("src.llm.backends.time.sleep", lambda seconds: None)

    backend = BigModelBackend(max_retries=3)
    payload = backend.generate_json("planner prompt")

    assert call_state["count"] == 2
    assert payload["ok"] is True


def test_bigmodel_backend_requires_bigmodel_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    backend = BigModelBackend()

    with pytest.raises(LLMBackendError, match="BIGMODEL_API_KEY"):
        backend.generate_text("hello")


def test_get_llm_constructs_bigmodel_backend():
    llm = get_llm(
        {
            "llm": {
                "provider": "bigmodel",
                "mode": "provider",
                "model": "glm-4.7-flashx",
                "api_key_env": "BIGMODEL_API_KEY",
            }
        }
    )

    assert isinstance(llm, BigModelBackend)
    assert llm.model == "glm-4.7-flashx"


def test_configuration_from_runtime_config_uses_bigmodel_defaults():
    cfg = Configuration.from_runtime_config({"llm": {"provider": "bigmodel", "mode": "provider"}})

    assert cfg.provider == "bigmodel"
    assert cfg.model == "glm-4.7-flashx"
    assert cfg.api_key_env == "BIGMODEL_API_KEY"
    assert cfg.base_url == "https://open.bigmodel.cn/api/paas/v4"
