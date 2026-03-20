from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_structure_docs_exist_and_debug_docs_are_archived():
    expected = [
        ROOT / "data/README.md",
        ROOT / "doc/structure/README.md",
        ROOT / "doc/structure/index.md",
        ROOT / "doc/structure/00_problem_and_protocol.md",
        ROOT / "doc/structure/01_dag_and_operators.md",
        ROOT / "doc/structure/02_workflow_and_bridge.md",
        ROOT / "doc/structure/03_training_and_evaluation.md",
        ROOT / "doc/structure/04_rebuild_checklist.md",
        ROOT / "doc/structure/05_missing_assets_and_roadmap.md",
        ROOT / "doc/structure/del/00_delete_policy.md",
        ROOT / "doc/structure/del/01_legacy_inventory.md",
        ROOT / "doc/archive/debug/operator_system_review.md",
    ]

    assert all(path.exists() for path in expected)
    assert not (ROOT / "doc/structure/debug").exists()


def test_legacy_layout_is_gone():
    assert not (ROOT / "src/tools").exists()
    assert not (ROOT / "src/graph").exists()
    assert not (ROOT / "src/cases").exists()


def test_structure_doc_content_contracts():
    data_readme = (ROOT / "data/README.md").read_text(encoding="utf-8")
    assert "Window Sample Contract" in data_readme
    assert "RM_101_THU_GEARBOX" in data_readme
    assert "RM_017_Ottawa19" in data_readme
    assert "window_id" in data_readme

    structure_readme = (ROOT / "doc/structure/README.md").read_text(encoding="utf-8")
    assert "根目录 `main.py` 是正式 Hydra 入口" in structure_readme
    assert "`scripts/preflight.py` 与 `scripts/run_case.py` 是由 `main.py` 调用的执行层库模块" in structure_readme
    assert "`config/runs/*.yaml` 是正式 Hydra preset 层" in structure_readme

    operators_doc = (ROOT / "doc/structure/01_dag_and_operators.md").read_text(encoding="utf-8")
    assert "torch+cu118" in operators_doc
    assert "np / pt / sym" in operators_doc
    assert "signal.wavefilters" in operators_doc

    workflow_doc = (ROOT / "doc/structure/02_workflow_and_bridge.md").read_text(encoding="utf-8")
    assert "validated DAG JSON" in workflow_doc
    assert "`workflow_state.json` 只保存状态快照与 `artifact_index_path`" in workflow_doc
    assert "runtime.workflow_mode=supervisor_proving" in workflow_doc
    assert "`plan -> execute -> compile -> verify`" in workflow_doc

    training_eval = (ROOT / "doc/structure/03_training_and_evaluation.md").read_text(encoding="utf-8")
    assert "deterministic / rule-based renderer" in training_eval
    assert "OfflineLLM.render_report()" in training_eval
    assert "split-level sampled dataset evidence" in training_eval
    assert "canonical diagnosis backend" in training_eval

    roadmap = (ROOT / "doc/structure/05_missing_assets_and_roadmap.md").read_text(encoding="utf-8")
    assert "M0: Agent Core" in roadmap
    assert "M1: Dataset-Level Evidence" in roadmap
    assert "M2: Comparison Layer" in roadmap
    assert "WaveFilters" in roadmap

    legacy_inventory = (ROOT / "doc/structure/del/01_legacy_inventory.md").read_text(encoding="utf-8")
    assert "`main.py` 单一入口 + `scripts/run_case.py` 库模块" in legacy_inventory
    assert "PHMState + StateGraph + chain-style agents" in legacy_inventory

    operator_review = (ROOT / "doc/archive/debug/operator_system_review.md").read_text(encoding="utf-8")
    assert "Accurate Findings" in operator_review
    assert "Needs Correction" in operator_review
    assert "multi-parent compiled support" in operator_review
