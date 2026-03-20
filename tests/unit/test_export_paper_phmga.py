from __future__ import annotations

import json
from pathlib import Path

import scripts.export_paper_phmga as export_module
from scripts.export_paper_phmga import _copy_artifacts, export_paper_phmga, parse_ledger, parse_worker_result


ROOT = Path(__file__).resolve().parents[2]


def test_parse_ledger_extracts_yaml_and_rows():
    ledger_meta, rows = parse_ledger()

    assert "active_stage_b_set" in ledger_meta
    assert "selected_global_best_backend" in ledger_meta
    assert rows
    assert any(row["experiment_id"] == "ottawa_ml_pilot_v1" for row in rows)
    assert any(row["experiment_id"] == "ottawa_ml_openrouter_glm_v1" for row in rows)


def test_parse_worker_result_extracts_checklists_and_sections():
    payload = parse_worker_result(ROOT / "doc/experiments/handoff/results/ottawa_ml_codex_v1.md")

    assert payload["metadata"]["experiment_id"] == "ottawa_ml_codex_v1"
    assert payload["artifact_checklist"]["validated_dag.json"] == "no"
    assert "Stage B" in payload["required_evidence"]["progress_record"]
    assert "Codex CLI transport" in payload["failure_summary"]


def test_export_paper_phmga_builds_self_contained_bundle(tmp_path: Path):
    export_paper_phmga(tmp_path)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    experiment_index = json.loads((tmp_path / "experiment_index.json").read_text(encoding="utf-8"))
    selection_status = json.loads((tmp_path / "selection_status.json").read_text(encoding="utf-8"))
    ledger_meta, rows = parse_ledger()

    assert manifest["experiment_count"] == len(rows)
    assert len(list((tmp_path / "subdocs").glob("*.md"))) == len(rows)
    assert len(experiment_index) == len(rows)
    assert selection_status["active_stage_b_set"] == ledger_meta["active_stage_b_set"]
    assert selection_status["selected_global_best_backend"] == ledger_meta["selected_global_best_backend"]
    assert "ottawa_ml_openrouter_v1" in selection_status["historical_failure_not_in_active_set"]

    pilot_subdoc = (tmp_path / "subdocs/ottawa_ml_pilot_v1.md").read_text(encoding="utf-8")
    assert pilot_subdoc.startswith("---\n")
    for section in (
        "# Summary",
        "# Status",
        "# Metrics",
        "# Artifacts",
        "# Feature / Diagnosis Evidence",
        "# Failure Or Pending Notes",
    ):
        assert section in pilot_subdoc

    pilot_evidence_dir = tmp_path / "evidence/ottawa_ml_pilot_v1"
    pilot_metadata = json.loads((pilot_evidence_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (pilot_evidence_dir / "artifacts/dag.json").exists()
    assert pilot_metadata["validated_dag_alias"] == "dag.json"

    failed_stage_b_dir = tmp_path / "evidence/ottawa_ml_codex_v1"
    failed_metadata = json.loads((failed_stage_b_dir / "metadata.json").read_text(encoding="utf-8"))
    assert (failed_stage_b_dir / "artifacts/planner_transport_trace.json").exists()
    assert (failed_stage_b_dir / "artifacts/planner_raw_response.txt").exists()
    assert failed_metadata["ledger_row"]["keep"] == "reject"

    pending_glm_subdoc = (tmp_path / "subdocs/ottawa_ml_openrouter_glm_v1.md").read_text(encoding="utf-8")
    assert "status: pending" in pending_glm_subdoc
    assert "z-ai/glm-4.5-air:free" in pending_glm_subdoc

    stage_b_doc = (tmp_path / "stages/stage_b_backend_comparison.md").read_text(encoding="utf-8")
    assert "historical_failure_not_in_active_set" in stage_b_doc
    assert "../subdocs/ottawa_ml_openrouter_glm_v1.md" in stage_b_doc


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
