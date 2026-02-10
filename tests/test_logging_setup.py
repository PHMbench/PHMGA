from __future__ import annotations

import json
from pathlib import Path

from src.utils.logging_setup import init_run_logger, log_event


def test_logging_setup_writes_jsonl_and_masks_secrets(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PHM_LOG_CONSOLE", "false")
    monkeypatch.setenv("PHM_LOG_JSON", "true")
    monkeypatch.setenv("PHM_LOG_FULL_LLM", "true")

    bundle = init_run_logger(case_name="unit_case", save_dir=tmp_path, run_id="run-001")
    log_event(
        bundle,
        level="INFO",
        event="llm.request",
        phase="builder",
        node="plan",
        message="unit test event",
        payload={
            "Authorization": "Bearer secret_token_1234567890",
            "api_key": "abcdef0123456789abcdef0123456789",
            "prompt": "hello",
        },
    )

    events_file = bundle.log_dir / "events.jsonl"
    run_log = bundle.log_dir / "run.log"
    assert events_file.exists()
    assert run_log.exists()

    lines = [ln for ln in events_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "events.jsonl should contain at least one event"
    event = json.loads(lines[-1])
    assert event["event"] == "llm.request"
    assert event["run_id"] == "run-001"
    assert event["case_name"] == "unit_case"
    assert event["payload"]["Authorization"] == "***"
    assert event["payload"]["api_key"] == "***"
    assert event["payload"]["prompt"] == "hello"

