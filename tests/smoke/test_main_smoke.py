from __future__ import annotations

from pathlib import Path

import yaml

from main import run_case


def test_with_report_smoke_run_emits_state_and_report(make_case_config):
    case = make_case_config(case_name="case_with_report", graph="with_report")

    payload = run_case(case["case_name"], config_root=case["config_root"])

    assert payload["status"] == "ok"
    assert payload["graph"] == "with_report"
    assert Path(case["state_save_path"]).exists()
    report_path = Path(case["report_path"])
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").strip()


def test_builder_then_executor_smoke_run_reuses_saved_state(make_case_config):
    case = make_case_config(case_name="case_builder_then_executor", graph="builder")

    builder_payload = run_case(case["case_name"], config_root=case["config_root"])

    assert builder_payload["status"] == "ok"
    assert builder_payload["graph"] == "builder"
    assert Path(case["state_save_path"]).exists()
    assert not Path(case["report_path"]).exists()

    config_path = Path(case["config_path"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["builder"]["graph"] = "executor"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    executor_payload = run_case(case["case_name"], config_root=case["config_root"])

    assert executor_payload["status"] == "ok"
    assert executor_payload["graph"] == "executor"
    report_path = Path(case["report_path"])
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").strip()
