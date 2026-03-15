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
    assert "scripts/preflight.py" in readme
    assert "scripts/run_case.py" in readme
    assert "config/runs/rm101_dag.yaml" in readme
    assert "mermaid" in readme
    assert "--dataset" not in readme

    for path in (ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / "GEMINI.md"):
        text = path.read_text(encoding="utf-8")
        assert "README.md" in text
        assert "main.py" not in text
        assert "src/tools" not in text
        assert "src/cases" not in text
