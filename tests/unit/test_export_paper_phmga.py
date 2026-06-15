from __future__ import annotations

import json
from pathlib import Path

import scripts.export_paper_phmga as export_module
from scripts.export_paper_phmga import (
    _copy_artifacts,
    _parse_provider_model,
    _stage_name_from_row,
    export_paper_phmga,
    parse_historical_incidents,
    parse_ledger,
    parse_worker_result,
)


ROOT = Path(__file__).resolve().parents[2]


def test_parse_ledger_extracts_yaml_and_rows():
    ledger_meta, rows = parse_ledger()

    assert "active_stage_b_set" in ledger_meta
    assert "selected_global_best_backend" in ledger_meta
    assert rows
    assert any(row["experiment_id"] == "ottawa_ml_openrouter_nemotron_v3" for row in rows)
    assert any(row["experiment_id"] == "rm101_ml_bigmodel_glm47_v1" for row in rows)
    assert all(
        _stage_name_from_row(row) == "stage_b_backend_comparison"
        for row in rows
        if row.get("run_type") == "backend_comparison"
    )


def test_parse_worker_result_extracts_checklists_and_sections():
    payload = parse_worker_result(ROOT / "doc/experiments/handoff/results/ottawa_ml_codex_v1.md")

    assert payload["metadata"]["experiment_id"] == "ottawa_ml_codex_v1"
    assert payload["artifact_checklist"]["validated_dag.json"] == "no"
    assert "Stage B" in payload["required_evidence"]["progress_record"]
    assert "Codex CLI transport" in payload["failure_summary"]


def test_parse_provider_model_strips_markdown_backticks():
    provider, model = _parse_provider_model("`codex_cli / gpt-5.3-codex`")

    assert provider == "codex_cli"
    assert model == "gpt-5.3-codex"


def test_export_paper_phmga_builds_self_contained_bundle(tmp_path: Path):
    protected_outline = tmp_path / "00_outline.md"
    protected_outline.write_text("outline must stay\n", encoding="utf-8")
    protected_thesis_dir = tmp_path / "thesis_rm101_case"
    protected_thesis_dir.mkdir()
    (protected_thesis_dir / "README.md").write_text("case asset must stay\n", encoding="utf-8")
    protected_historical_dir = tmp_path / "evidence/historical_keep"
    protected_historical_dir.mkdir(parents=True)
    (protected_historical_dir / "metadata.json").write_text("{}", encoding="utf-8")

    export_paper_phmga(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    experiment_index = json.loads((tmp_path / "experiment_index.json").read_text(encoding="utf-8"))
    selection_status = json.loads((tmp_path / "selection_status.json").read_text(encoding="utf-8"))
    historical_incidents = json.loads((tmp_path / "historical_incidents.json").read_text(encoding="utf-8"))
    ledger_meta, rows = parse_ledger()

    assert manifest["experiment_count"] == len(rows)
    assert len(list((tmp_path / "subdocs").glob("*.md"))) == len(rows)
    assert len(experiment_index) == len(rows)
    assert selection_status["active_stage_b_set"] == ledger_meta["active_stage_b_set"]
    assert selection_status["selected_global_best_backend"] == ledger_meta["selected_global_best_backend"]
    assert protected_outline.read_text(encoding="utf-8") == "outline must stay\n"
    assert (protected_thesis_dir / "README.md").read_text(encoding="utf-8") == "case asset must stay\n"
    assert (protected_historical_dir / "metadata.json").exists()
    assert any(item["experiment_id"] == "ottawa_ml_openrouter_v1" for item in historical_incidents)
    assert any(item["target"] == "doc/experiments/incidents/03_openrouter_api_analysis.md" for item in historical_incidents)

    accepted_subdoc = (tmp_path / "subdocs/ottawa_ml_openrouter_nemotron_v3.md").read_text(encoding="utf-8")
    assert accepted_subdoc.startswith("---\n")
    for section in (
        "# Summary",
        "# Status",
        "# Metrics",
        "# Artifacts",
        "# Feature / Diagnosis Evidence",
        "# Failure Or Pending Notes",
    ):
        assert section in accepted_subdoc

    accepted_evidence_dir = tmp_path / "evidence/ottawa_ml_openrouter_nemotron_v3"
    accepted_metadata = json.loads((accepted_evidence_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (accepted_evidence_dir / "artifacts/validated_dag.json").exists()
    assert accepted_metadata["ledger_row"]["keep"] == "accept"
    assert accepted_metadata["derived_stage_name"] == "stage_b_backend_comparison"

    failed_stage_b_dir = tmp_path / "evidence/rm101_ml_openrouter_nemotron_v3"
    failed_metadata = json.loads((failed_stage_b_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (failed_stage_b_dir / "artifacts/validated_dag.json").exists()
    assert (failed_stage_b_dir / "artifacts/workflow_state.json").exists()
    assert failed_metadata["ledger_row"]["keep"] == "reject"
    assert failed_metadata["derived_stage_name"] == "stage_b_backend_comparison"

    pending_glm_subdoc = (tmp_path / "subdocs/rm101_ml_openrouter_glm_v2.md").read_text(encoding="utf-8")
    assert "status: pending" in pending_glm_subdoc
    assert "z-ai/glm-4.5-air:free" in pending_glm_subdoc

    pending_codex_subdoc = (tmp_path / "subdocs/ottawa_ml_codex_v3.md").read_text(encoding="utf-8")
    assert "model: gpt-5.3-codex\n" in pending_codex_subdoc

    stage_b_doc = (tmp_path / "stages/stage_b_backend_comparison.md").read_text(encoding="utf-8")
    assert "active_bigmodel" in stage_b_doc
    assert "../subdocs/ottawa_ml_openrouter_nemotron_v3.md" in stage_b_doc
    assert "../subdocs/rm101_ml_bigmodel_glm47_v1.md" in stage_b_doc

    historical_doc = (tmp_path / "historical_incidents.md").read_text(encoding="utf-8")
    assert "must not enter main tables" in historical_doc
    assert "ottawa_ml_openrouter_v1" in historical_doc


def test_parse_historical_incidents_keeps_failure_traceability_outside_selection():
    incidents = parse_historical_incidents()

    assert any(item["experiment_id"] == "rm101_ml_openrouter_v1" for item in incidents)
    assert any(item["target"] == "doc/experiments/incidents/03_openrouter_api_analysis.md" for item in incidents)
    assert all("formal_selection" in item["selection_scope"] for item in incidents)


def test_copy_artifacts_accepts_current_validated_dag_only_layout(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(export_module, "ROOT", tmp_path)
    source_dir = tmp_path / "runtime_artifacts"
    source_dir.mkdir()
    (source_dir / "validated_dag.json").write_text("{}", encoding="utf-8")
    (source_dir / "artifact_index.json").write_text("{}", encoding="utf-8")
    (source_dir / "workflow_state.json").write_text('{"artifact_index_path":"artifact_index.json"}', encoding="utf-8")

    evidence_dir = tmp_path / "bundle_evidence"
    evidence_dir.mkdir()
    artifact_manifest = _copy_artifacts({"output_dir": "runtime_artifacts"}, evidence_dir)

    assert artifact_manifest["artifact_presence"]["validated_dag.json"] is True
    assert artifact_manifest["artifact_presence"]["dag.json"] is False
    assert artifact_manifest["validated_dag_alias"] is None
    assert (evidence_dir / "artifacts/validated_dag.json").exists()
