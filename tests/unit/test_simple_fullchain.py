from __future__ import annotations

from pathlib import Path

import pytest

import src.phm_outer_graph as phm_outer_graph_module
from scripts import run_case as run_case_module
from src.config import load_runtime_config
from src.evaluation import evaluate_artifact_contract


ROOT = Path(__file__).resolve().parents[2]


def _simple_runtime_config(tmp_path: Path) -> dict:
    runtime_config = load_runtime_config(
        ROOT / "config/runs/ottawa_synth_ml_simple.yaml",
        output_dir=str(tmp_path / "ottawa_synth_simple_fullchain"),
    )
    runtime_config["llm"]["mode"] = "offline_stub"
    return runtime_config


def test_run_case_simple_fullchain_skips_dag_quality_and_llm_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime_config = _simple_runtime_config(tmp_path)

    def _unexpected(*args, **kwargs):
        raise AssertionError("simple_fullchain should not call rich-only helpers")

    monkeypatch.setattr(phm_outer_graph_module, "build_dag_quality_summary", _unexpected)
    monkeypatch.setattr(phm_outer_graph_module, "report_agent", _unexpected)

    payload = run_case_module.run_case(runtime_config)
    output_dir = Path(payload["output_dir"])

    assert payload["graph_path"] == "ml"
    assert evaluate_artifact_contract(output_dir) is True
    assert (output_dir / "validated_dag.json").exists()
    assert (output_dir / "compiled_dag_manifest.json").exists()
    assert (output_dir / "feature_pipeline.json").exists()
    assert (output_dir / "feature_list.json").exists()
    assert (output_dir / "feature_separability_summary.json").exists()
    assert (output_dir / "artifact_index.json").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "final_report.md").exists()
    assert not (output_dir / "dag_quality_summary.json").exists()

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert report_text.startswith("# PHMGA Final Report:")
