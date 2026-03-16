from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_structure_docs_exist():
    expected = [
        ROOT / "doc/structure/README.md",
        ROOT / "doc/structure/index.md",
        ROOT / "doc/structure/00_problem_and_protocol.md",
        ROOT / "doc/structure/01_dag_and_operators.md",
        ROOT / "doc/structure/02_workflow_and_bridge.md",
        ROOT / "doc/structure/03_training_and_evaluation.md",
        ROOT / "doc/structure/04_rebuild_checklist.md",
        ROOT / "doc/structure/del/00_delete_policy.md",
        ROOT / "doc/structure/del/01_legacy_inventory.md",
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
    assert "--dataset" not in readme

    operators_doc = (ROOT / "doc/structure/01_dag_and_operators.md").read_text(encoding="utf-8")
    assert "torch+cu118" in operators_doc
    assert "np / pt / sym" in operators_doc
    assert "不是外部 `conda` 环境" in operators_doc

    training_eval = (ROOT / "doc/structure/03_training_and_evaluation.md").read_text(encoding="utf-8")
    assert "deterministic / rule-based renderer" in training_eval
    assert "OfflineLLM.render_report()" in training_eval
    assert "torch+cu118" in training_eval
    assert "不是外部 `conda` 环境" in training_eval

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
