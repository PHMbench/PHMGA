from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_experiment_docs_exist_and_legacy_protocol_docs_are_gone():
    expected = [
        ROOT / "doc/experiments/00_manual_runbook.md",
        ROOT / "doc/experiments/01_result_ledger.md",
        ROOT / "doc/experiments/02_main_tables.md",
        ROOT / "doc/experiments/04_execution_protocol.md",
        ROOT / "doc/experiments/05_worker_result_template.md",
        ROOT / "doc/experiments/incidents/03_openrouter_api_analysis.md",
        ROOT / "doc/experiments/handoff/01_pilot_owner.md",
        ROOT / "doc/experiments/handoff/02_main_owner_ottawa.md",
        ROOT / "doc/experiments/handoff/03_main_owner_rm101.md",
        ROOT / "doc/experiments/handoff/04_ablation_owner_ml.md",
        ROOT / "doc/experiments/handoff/05_ablation_owner_torch.md",
        ROOT / "doc/experiments/handoff/06_backend_comparison_owner.md",
        ROOT / "doc/experiments/handoff/07_harness_engineer.md",
        ROOT / "doc/experiments/handoff/results/README.md",
    ]

    assert all(path.exists() for path in expected)
    assert not (ROOT / "doc/experiments/03_openrouter_api_analysis.md").exists()
    assert not (ROOT / "doc/experiments/04_codex_cli_handoff.md").exists()
    assert not (ROOT / "doc/experiments/06_multi_agent_merge_checklist.md").exists()


def test_experiment_doc_content_contracts():
    runbook = (ROOT / "doc/experiments/00_manual_runbook.md").read_text(encoding="utf-8")
    assert "Backend Comparison On Canonical ML Mainline" in runbook
    assert "04_execution_protocol.md" in runbook
    assert "05_worker_result_template.md" in runbook
    assert "Artifact Contract Gate" in runbook
    assert "Feature Separability Gate" in runbook
    assert "validated_dag.json" in runbook
    assert "artifact_index.json" in runbook
    assert "final_report.md" in runbook
    assert "06_multi_agent_merge_checklist.md" not in runbook

    ledger = (ROOT / "doc/experiments/01_result_ledger.md").read_text(encoding="utf-8")
    assert "active_stage_b_set" in ledger
    assert "selected_global_best_backend" in ledger
    assert "artifact_contract_pass" in ledger
    assert "feature_separability_pass" in ledger
    assert "selection_eligible" in ledger

    main_tables = (ROOT / "doc/experiments/02_main_tables.md").read_text(encoding="utf-8")
    assert "Backend Comparison And Selection" in main_tables
    assert "Best-Backend Ablations" in main_tables
    assert "ottawa_ml_openrouter_glm_v1" in main_tables
    assert "selected_global_best_backend" in main_tables

    execution_protocol = (ROOT / "doc/experiments/04_execution_protocol.md").read_text(encoding="utf-8")
    assert "worker tool = Codex CLI" in execution_protocol
    assert "experiment backend = active/selected backend tuple" in execution_protocol
    assert "results/<experiment_id>.md" in execution_protocol
    assert "OPENROUTER_API_KEY" in execution_protocol
    assert "artifact_contract_pass" in execution_protocol
    assert "feature_separability_pass" in execution_protocol
    assert "accept" in execution_protocol
    assert "reject" in execution_protocol
    assert "needs_rerun" in execution_protocol

    worker_template = (ROOT / "doc/experiments/05_worker_result_template.md").read_text(encoding="utf-8")
    assert "本文件只保留模板和字段规则" in worker_template
    assert "04_execution_protocol.md" in worker_template
    assert "validated_dag.json" in worker_template
    assert "artifact_index.json" in worker_template
    assert "feature_separability_summary" in worker_template
    assert "progress_record" in worker_template
    assert "## Authority" not in worker_template

    results_readme = (ROOT / "doc/experiments/handoff/results/README.md").read_text(encoding="utf-8")
    assert "results/<experiment_id>.md" in results_readme
    assert "feature_list" in results_readme
    assert "progress_record" in results_readme

    comparison_ticket = (ROOT / "doc/experiments/handoff/06_backend_comparison_owner.md").read_text(encoding="utf-8")
    assert "ottawa_ml_openrouter_glm_v1" in comparison_ticket
    assert "rm101_ml_openrouter_glm_v1" in comparison_ticket
    assert "only modify" not in comparison_ticket.lower()

    harness_ticket = (ROOT / "doc/experiments/handoff/07_harness_engineer.md").read_text(encoding="utf-8")
    assert "artifact_contract_pass" in harness_ticket
    assert "feature_separability_pass" in harness_ticket
    assert "selection_eligible" in harness_ticket

    incident_note = (ROOT / "doc/experiments/incidents/03_openrouter_api_analysis.md").read_text(encoding="utf-8")
    assert "OpenRouter Candidate Note" in incident_note
    assert "comparison candidate" in incident_note
    assert "selected_global_best_backend" in incident_note
    assert "historical" in incident_note

    example_summary = json.loads(
        (ROOT / "doc/experiments/examples/feature_separability_summary.example.json").read_text(encoding="utf-8")
    )
    assert "top_features" in example_summary
    assert "aggregate_scores" in example_summary
    assert "split_stability" in example_summary
    assert "decision" in example_summary
    assert "reason" in example_summary
