from __future__ import annotations

from pathlib import Path

import pytest

import src.phm_outer_graph as phm_outer_graph_module
from scripts import run_case as run_case_module
from scripts.preflight import run_preflight
from src.config import load_runtime_config
from src.evaluation import REQUIRED_STAGE_B_ARTIFACTS, evaluate_artifact_contract
from src.operators import SUPERVISOR_PROVING_PLAN_NAMES, get_supervisor_proving_catalog


ROOT = Path(__file__).resolve().parents[2]


def _proving_runtime_config(tmp_path: Path) -> dict:
    runtime_config = load_runtime_config(
        ROOT / "config/runs/ottawa_synth_ml.yaml",
        output_dir=str(tmp_path / "ottawa_synth_supervisor_proving"),
    )
    runtime_config["runtime"]["workflow_mode"] = "supervisor_proving"
    runtime_config["runtime"]["max_iterations"] = 1
    runtime_config["evaluation"]["dag_quality"]["enabled"] = False
    runtime_config["llm"]["mode"] = "offline_stub"
    return runtime_config


def test_supervisor_proving_catalog_exposes_only_strict_deterministic_ops():
    catalog = get_supervisor_proving_catalog()
    summary = catalog.summary()

    assert {row["op_name"] for row in summary} == set(SUPERVISOR_PROVING_PLAN_NAMES)
    assert all(not row["llm_tunable_params"] for row in summary)
    assert all(row["schema_category"] not in {"MULTI_VARIABLE", "DECISION"} for row in summary)


def test_run_case_supervisor_proving_uses_light_graph_and_writes_auditable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _proving_runtime_config(tmp_path)

    def _unexpected(*args, **kwargs):
        raise AssertionError("rich-graph-only helper should not be called in supervisor proving mode")

    monkeypatch.setattr(phm_outer_graph_module, "build_dag_quality_summary", _unexpected)
    monkeypatch.setattr(phm_outer_graph_module, "reflect_agent", _unexpected)
    monkeypatch.setattr(phm_outer_graph_module, "report_agent", _unexpected)
    monkeypatch.setattr(phm_outer_graph_module, "inquirer_agent", _unexpected)

    payload = run_case_module.run_case(runtime_config)
    output_dir = Path(payload["output_dir"])

    assert payload["graph_path"] == "ml"
    assert evaluate_artifact_contract(output_dir) is True
    assert (output_dir / "step_plan.json").exists()
    assert (output_dir / "validated_dag.json").exists()
    assert (output_dir / "compiled_dag_manifest.json").exists()
    assert not (output_dir / "dag_quality_summary.json").exists()

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert report_text.startswith("# PHMGA Final Report:")
    for artifact_name in REQUIRED_STAGE_B_ARTIFACTS:
        assert (output_dir / artifact_name).exists()


def test_preflight_requires_openrouter_key_for_provider_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime_config = load_runtime_config(ROOT / "config/runs/ottawa_synth_ml.yaml")
    runtime_config["llm"]["mode"] = "provider"
    runtime_config["llm"]["provider"] = "openrouter"
    runtime_config["llm"]["model"] = "z-ai/glm-4.5-air:free"
    runtime_config["llm"]["api_key_env"] = "OPENROUTER_API_KEY"
    runtime_config["llm"]["env_file"] = str(tmp_path / "missing.env")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        run_preflight(runtime_config)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    payload = run_preflight(runtime_config)

    assert payload["status"] == "ok"
    assert payload["llm_transport"]["provider"] == "openrouter"
    assert payload["llm_transport"]["credential_found"] is True
    assert payload["llm_transport"]["api_key_env"] == "OPENROUTER_API_KEY"


def test_preflight_requires_bigmodel_key_for_provider_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime_config = load_runtime_config(ROOT / "config/runs/ottawa_synth_ml.yaml")
    runtime_config["llm"]["mode"] = "provider"
    runtime_config["llm"]["provider"] = "bigmodel"
    runtime_config["llm"]["model"] = "glm-4.7-flash"
    runtime_config["llm"]["api_key_env"] = "BIGMODEL_API_KEY"
    runtime_config["llm"]["env_file"] = str(tmp_path / "missing.env")

    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BIGMODEL_API_KEY"):
        run_preflight(runtime_config)

    monkeypatch.setenv("BIGMODEL_API_KEY", "test-bigmodel-key")
    payload = run_preflight(runtime_config)

    assert payload["status"] == "ok"
    assert payload["llm_transport"]["provider"] == "bigmodel"
    assert payload["llm_transport"]["credential_found"] is True
    assert payload["llm_transport"]["api_key_env"] == "BIGMODEL_API_KEY"


def test_preflight_loads_provider_key_from_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime_config = load_runtime_config(ROOT / "config/runs/ottawa_synth_ml.yaml")
    runtime_config["llm"]["mode"] = "provider"
    runtime_config["llm"]["provider"] = "bigmodel"
    runtime_config["llm"]["model"] = "glm-4.7-flash"
    runtime_config["llm"]["api_key_env"] = "BIGMODEL_API_KEY"
    runtime_config["llm"]["env_file"] = str(tmp_path / ".env")
    (tmp_path / ".env").write_text("BIGMODEL_API_KEY=test-bigmodel-key\n", encoding="utf-8")

    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    payload = run_preflight(runtime_config)

    assert payload["status"] == "ok"
    assert payload["llm_transport"]["provider"] == "bigmodel"
    assert payload["llm_transport"]["credential_found"] is True
