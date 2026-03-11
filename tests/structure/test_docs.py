from __future__ import annotations

from pathlib import Path


DOC_ROOT = Path(__file__).resolve().parents[2] / "doc" / "structure"
PLAN_ROOT = DOC_ROOT.parent / "plan"


def test_docs_module_contains_structure_index_and_sections():
    required = [
        DOC_ROOT / "README.md",
        DOC_ROOT / "index.md",
        DOC_ROOT / "config" / "structure.md",
        DOC_ROOT / "data" / "structure.md",
        DOC_ROOT / "graph" / "structure.md",
        DOC_ROOT / "llm" / "structure.md",
        DOC_ROOT / "agents" / "structure.md",
        DOC_ROOT / "cases" / "structure.md",
        DOC_ROOT / "model" / "structure.md",
        DOC_ROOT / "tools" / "structure.md",
        DOC_ROOT / "schemas" / "structure.md",
        DOC_ROOT / "prompts" / "structure.md",
        DOC_ROOT / "states" / "structure.md",
        DOC_ROOT / "utils" / "structure.md",
        DOC_ROOT / "docs" / "structure.md",
        DOC_ROOT / "del" / "README.md",
        DOC_ROOT / "del" / "00_redundancy_inventory.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing


def test_docs_module_records_redundancy_inventory():
    inventory = (DOC_ROOT / "del" / "00_redundancy_inventory.md").read_text(encoding="utf-8")
    assert "src/utils.py" in inventory
    assert "src/phm_outer_graph.py" in inventory
    assert "src/model.py" in inventory
    assert "case1 -> data_cfg" in inventory
    assert "ref/test" in inventory
    assert "src/agents/shared/compat.py" in inventory
    assert "__pycache__" in inventory


def test_docs_module_structure_pages_explain_why_layer_exists():
    required_markers = [
        "## 职责",
        "## 为什么需要这一层",
        "## 正式入口",
        "## 当前实现状态",
        "## 冗余与历史包袱",
    ]
    for path in sorted(DOC_ROOT.glob("*/structure.md")):
        text = path.read_text(encoding="utf-8")
        for marker in required_markers:
            assert marker in text, f"{path} missing section: {marker}"


def test_docs_index_describes_boundaries_not_just_links():
    text = (DOC_ROOT / "index.md").read_text(encoding="utf-8")
    assert "为什么必须独立成层" in text
    assert "与相邻层的边界" in text
    assert "train/val/test" in text


def test_agents_readme_uses_canonical_split_terms():
    text = (DOC_ROOT.parents[1] / "src" / "agents" / "readme.md").read_text(encoding="utf-8")
    assert "labels_ref" not in text
    assert "labels_tst" not in text
    assert "train/val/test" in text


def test_agents_structure_documents_agent_contracts():
    text = (DOC_ROOT / "agents" / "structure.md").read_text(encoding="utf-8")
    required_markers = [
        "## 分类原则",
        "### LLM-mediated agents",
        "### Deterministic agents",
        "输入",
        "输出",
        "失败方式",
    ]
    required_agents = [
        "plan_agent",
        "dag_init_agent",
        "execute_agent",
        "reflect_agent",
        "report_agent",
        "deep_research_agents",
        "dataset_preparer_agent",
        "deep_model_train_agent",
        "tspn_bootstrap_agent",
        "inquirer_agent",
        "shallow_ml_agent",
    ]
    for marker in required_markers:
        assert marker in text
    for agent_name in required_agents:
        assert f"`{agent_name}`" in text


def test_plan_docs_do_not_embed_project_artifacts_or_empty_dirs():
    assert not any(PLAN_ROOT.rglob(".env"))
    assert not any(PLAN_ROOT.rglob(".idea"))
    assert not any(PLAN_ROOT.rglob("__pycache__"))
    empty_dirs = [path for path in PLAN_ROOT.rglob("*") if path.is_dir() and not any(path.iterdir())]
    assert not empty_dirs
