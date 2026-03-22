from __future__ import annotations

import pytest

from src.llm.backends import OpenRouterBackend


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
