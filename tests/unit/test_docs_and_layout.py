from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_structure_docs_exist():
    expected = [
        ROOT / "data/README.md",
        ROOT / "doc/structure/README.md",
        ROOT / "doc/structure/index.md",
        ROOT / "doc/structure/00_problem_and_protocol.md",
        ROOT / "doc/structure/01_dag_and_operators.md",
        ROOT / "doc/structure/02_workflow_and_bridge.md",
        ROOT / "doc/structure/03_training_and_evaluation.md",
        ROOT / "doc/structure/04_rebuild_checklist.md",
        ROOT / "doc/structure/del/00_delete_policy.md",
        ROOT / "doc/structure/del/01_legacy_inventory.md",
        ROOT / "doc/ablation/00_master_plan.md",
        ROOT / "doc/ablation/01_graphmodule_runtime_matrix.md",
        ROOT / "doc/ablation/02_wavefilters_ablation.md",
        ROOT / "doc/ablation/03_learnable_control_ablation.md",
        ROOT / "doc/experiments/00_manual_runbook.md",
        ROOT / "doc/experiments/01_result_ledger.md",
        ROOT / "doc/experiments/02_main_tables.md",
        ROOT / "doc/experiments/04_codex_cli_handoff.md",
        ROOT / "doc/experiments/05_worker_result_template.md",
        ROOT / "doc/experiments/06_multi_agent_merge_checklist.md",
        ROOT / "doc/experiments/handoff/01_pilot_owner.md",
        ROOT / "doc/experiments/handoff/02_main_owner_ottawa.md",
        ROOT / "doc/experiments/handoff/03_main_owner_rm101.md",
        ROOT / "doc/experiments/handoff/04_ablation_owner_ml.md",
        ROOT / "doc/experiments/handoff/05_ablation_owner_torch.md",
        ROOT / "doc/experiments/handoff/06_backend_comparison_owner.md",
        ROOT / "doc/experiments/handoff/07_harness_engineer.md",
        ROOT / "doc/experiments/handoff/results/README.md",
        ROOT / "doc/paper/00_outline.md",
        ROOT / "config/runs/rm101_ml_test.yaml",
        ROOT / "config/runs/rm101_torch_test.yaml",
    ]
    assert all(path.exists() for path in expected)


def test_legacy_layout_is_gone():
    assert not (ROOT / "src/tools").exists()
    assert not (ROOT / "src/graph").exists()
    assert not (ROOT / "src/cases").exists()


def test_ai_guides_reference_readme_and_drop_legacy_paths():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "<<<<<<<" not in readme
    assert "dag_only" in readme
    assert "main.py" in readme
    assert "runtime.action=preflight" in readme
    assert "+runs=rm101_dag" in readme
    assert "config/runs/rm101_dag.yaml" in readme
    assert "mermaid" in readme
    assert "dag_quality_evaluator" in readme
    assert "deterministic / rule-based renderer" in readme
    assert "torch+cu118" in readme
    assert "不是外部 `conda` 环境" in readme
    assert "mode=provider" in readme
    assert "formal main presets" in readme
    assert "codex_cli" in readme
    assert "gpt-5.3-codex" in readme
    assert "offline_stub" in readme
    assert "smoke/development baseline" in readme
    assert "split-level sampled dataset evidence" in readme
    assert "M0: Agent Core" in readme
    assert "M1: Dataset-Level Evidence" in readme
    assert "M2: Comparison Layer" in readme
    assert "canonical diagnosis backend" in readme
    assert "--dataset" not in readme

    data_readme = (ROOT / "data/README.md").read_text(encoding="utf-8")
    assert "Window Sample Contract" in data_readme
    assert "RM_101_THU_GEARBOX" in data_readme
    assert "RM_017_Ottawa19" in data_readme
    assert "window_id" in data_readme
    assert "source_sample_id" in data_readme

    operators_doc = (ROOT / "doc/structure/01_dag_and_operators.md").read_text(encoding="utf-8")
    assert "torch+cu118" in operators_doc
    assert "np / pt / sym" in operators_doc
    assert "不是外部 `conda` 环境" in operators_doc
    assert "signal.wavefilters" in operators_doc

    training_eval = (ROOT / "doc/structure/03_training_and_evaluation.md").read_text(encoding="utf-8")
    assert "deterministic / rule-based renderer" in training_eval
    assert "OfflineLLM.render_report()" in training_eval
    assert "torch+cu118" in training_eval
    assert "不是外部 `conda` 环境" in training_eval
    assert "provider mode" in training_eval or "provider-backed" in training_eval
    assert "channel_self_attention" in training_eval
    assert "split-level sampled dataset evidence" in training_eval
    assert "比较层" in training_eval
    assert "M0: Agent Core" in training_eval
    assert "canonical diagnosis backend" in training_eval

    roadmap = (ROOT / "doc/structure/05_missing_assets_and_roadmap.md").read_text(encoding="utf-8")
    assert "WaveFilters" in roadmap
    assert "GraphModule" in roadmap
    assert "Codex" in roadmap
    assert "split-level sampled evidence pass" in roadmap
    assert "M0: Agent Core" in roadmap
    assert "M1: Dataset-Level Evidence" in roadmap
    assert "M2: Comparison Layer" in roadmap

    ablation_master = (ROOT / "doc/ablation/00_master_plan.md").read_text(encoding="utf-8")
    assert "compiled" in ablation_master
    assert "module_runtime" in ablation_master
    assert "doc/experiments/01_result_ledger.md" in ablation_master

    runbook = (ROOT / "doc/experiments/00_manual_runbook.md").read_text(encoding="utf-8")
    assert "ottawa_ml_test" in runbook
    assert "ottawa_torch_test" in runbook
    assert "rm101_ml_test" in runbook
    assert "rm101_torch_test" in runbook
    assert "doc/experiments/01_result_ledger.md" in runbook
    assert "Backend Comparison On Canonical ML Mainline" in runbook
    assert "llm.mode=provider" in runbook
    assert "llm.provider=codex_cli" in runbook
    assert "gpt-5.3-codex" in runbook
    assert "smoke/development baseline" in runbook
    assert "04_codex_cli_handoff.md" in runbook
    assert "05_worker_result_template.md" in runbook
    assert "06_multi_agent_merge_checklist.md" in runbook
    assert "results/<experiment_id>.md" in runbook
    assert "Research Closure Milestones" in runbook
    assert "canonical diagnosis backend" in runbook
    assert "Artifact Contract Gate" in runbook
    assert "Feature Separability Gate" in runbook
    assert "selected_global_best_backend" in runbook
    assert "artifact_index.json" in runbook

    ledger = (ROOT / "doc/experiments/01_result_ledger.md").read_text(encoding="utf-8")
    assert "active_stage_b_set" in ledger
    assert "selected_global_best_backend" in ledger
    assert "artifact_contract_pass" in ledger
    assert "feature_separability_pass" in ledger
    assert "selection_eligible" in ledger
    assert "selection_basis" in ledger
    assert "backend comparison candidate" in ledger
    assert "formal main using selected_global_best_backend" in ledger
    assert "ablation on selected_global_best_backend" in ledger

    handoff = (ROOT / "doc/experiments/04_codex_cli_handoff.md").read_text(encoding="utf-8")
    assert "Codex CLI worker" in handoff
    assert "codex login" in handoff
    assert "OPENROUTER_API_KEY" in handoff
    assert "不得改 provider 默认" in handoff
    assert "01_pilot_owner.md" in handoff
    assert "06_backend_comparison_owner.md" in handoff
    assert "07_harness_engineer.md" in handoff
    assert "先写 worker 结果报告" in handoff
    assert "results/<experiment_id>.md" in handoff
    assert "worker tool = Codex CLI" in handoff
    assert "experiment backend = active/selected backend tuple" in handoff
    assert "harness engineer" in handoff

    worker_template = (ROOT / "doc/experiments/05_worker_result_template.md").read_text(encoding="utf-8")
    assert "worker_id" in worker_template
    assert "experiment_id" in worker_template
    assert "ledger_updated" in worker_template
    assert "先写 `results/<experiment_id>.md`" in worker_template
    assert "artifact_contract_pass" in worker_template
    assert "feature_separability_pass" in worker_template
    assert "feature_list" in worker_template
    assert "feature_separability_summary" in worker_template
    assert "progress_record" in worker_template
    assert "artifact_index.json" in worker_template

    merge_doc = (ROOT / "doc/experiments/06_multi_agent_merge_checklist.md").read_text(encoding="utf-8")
    assert "accept" in merge_doc
    assert "reject" in merge_doc
    assert "needs_rerun" in merge_doc
    assert "02_main_tables.md" in merge_doc
    assert "selected_global_best_backend" in merge_doc
    assert "artifact_contract_pass" in merge_doc
    assert "feature_separability_pass" in merge_doc
    assert "harness engineer" in merge_doc

    results_readme = (ROOT / "doc/experiments/handoff/results/README.md").read_text(encoding="utf-8")
    assert "results/<experiment_id>.md" in results_readme
    assert "01_result_ledger.md" in results_readme
    assert "feature_list" in results_readme
    assert "progress_record" in results_readme
    assert "runtime-native artifact" in results_readme

    comparison_ticket = (ROOT / "doc/experiments/handoff/06_backend_comparison_owner.md").read_text(encoding="utf-8")
    assert "results/ottawa_ml_openrouter_v1.md" in comparison_ticket
    assert "results/ottawa_ml_codex_v1.md" in comparison_ticket
    assert "01_result_ledger.md" in comparison_ticket

    harness_ticket = (ROOT / "doc/experiments/handoff/07_harness_engineer.md").read_text(encoding="utf-8")
    assert "artifact_contract_pass" in harness_ticket
    assert "feature_separability_pass" in harness_ticket
    assert "selection_eligible" in harness_ticket
    assert "artifact_index.json" in harness_ticket

    torch_ablation_ticket = (ROOT / "doc/experiments/handoff/05_ablation_owner_torch.md").read_text(encoding="utf-8")
    assert "results/ottawa_torch_module_runtime_v1.md" in torch_ablation_ticket
    assert "01_result_ledger.md" in torch_ablation_ticket
    assert "selected_global_best_backend" in torch_ablation_ticket

    main_tables = (ROOT / "doc/experiments/02_main_tables.md").read_text(encoding="utf-8")
    assert "ottawa_ml_main_v1" in main_tables
    assert "rm101_torch_attention_v1" in main_tables
    assert "doc/experiments/01_result_ledger.md" in main_tables
    assert "Backend Comparison And Selection" in main_tables
    assert "Best-Backend Ablations" in main_tables
    assert "gpt-5.3-codex" in main_tables
    assert "ottawa_ml_codex_v1" in main_tables
    assert "path comparison" in main_tables
    assert "canonical diagnosis mainline" in main_tables
    assert "selected_global_best_backend" in main_tables

    paper_outline = (ROOT / "doc/paper/00_outline.md").read_text(encoding="utf-8")
    assert "PHMState" in paper_outline
    assert "StateGraph" in paper_outline
    assert "multi-parent compiled support" in paper_outline
    assert "WaveFilters" in paper_outline

    legacy_inventory = (ROOT / "doc/structure/del/01_legacy_inventory.md").read_text(encoding="utf-8")
    assert "PHMState + StateGraph + chain-style agents" in legacy_inventory

    openrouter_note = (ROOT / "doc/experiments/03_openrouter_api_analysis.md").read_text(encoding="utf-8")
    assert "text-mode" in openrouter_note
    assert "formal main" in openrouter_note
    assert "comparison candidate" in openrouter_note
    assert "selected_global_best_backend" in openrouter_note

    for path in (
        ROOT / "config/runs/ottawa_ml.yaml",
        ROOT / "config/runs/ottawa_torch.yaml",
        ROOT / "config/runs/rm101_ml.yaml",
        ROOT / "config/runs/rm101_torch.yaml",
    ):
        text = path.read_text(encoding="utf-8")
        assert "provider: codex_cli" in text
        assert "gpt-5.3-codex" in text

    operator_review = (ROOT / "doc/structure/debug/operator_system_review.md").read_text(encoding="utf-8")
    assert "Accurate Findings" in operator_review
    assert "Needs Correction" in operator_review
    assert "multi-parent compiled support" in operator_review
    assert "BaseIsomorphicOperator" in operator_review
    assert "BaseModel" in operator_review

    for path in (ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / "GEMINI.md"):
        text = path.read_text(encoding="utf-8")
        assert "README.md" in text
        assert "main.py" in text
        assert "src/tools" not in text
        assert "src/cases" not in text
