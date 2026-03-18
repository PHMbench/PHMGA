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
    assert "offline_stub" in readme
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

    roadmap = (ROOT / "doc/structure/05_missing_assets_and_roadmap.md").read_text(encoding="utf-8")
    assert "WaveFilters" in roadmap
    assert "GraphModule" in roadmap

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
    assert "provider-backed LLM" in runbook
    assert "llm.mode=provider" in runbook

    main_tables = (ROOT / "doc/experiments/02_main_tables.md").read_text(encoding="utf-8")
    assert "ottawa_ml_main_v1" in main_tables
    assert "rm101_torch_attention_v1" in main_tables
    assert "doc/experiments/01_result_ledger.md" in main_tables
    assert "| provider |" in main_tables

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
