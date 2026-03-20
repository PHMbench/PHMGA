from __future__ import annotations

from pathlib import Path

import pytest

from src.config import compose_runtime_config, to_runtime_dict


ROOT = Path(__file__).resolve().parents[2]


def test_public_entry_contract_uses_main_py_and_library_scripts():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "main.py" in readme
    assert "runtime.action=preflight" in readme
    assert "config/config.yaml" in readme
    assert "config/runs/*.yaml" in readme
    assert "python scripts/preflight.py" not in readme
    assert "python scripts/run_case.py" not in readme
    assert "执行层 `scripts/*.py` 不再作为公共 CLI 入口" in readme

    for path in (ROOT / "AGENTS.md", ROOT / "CLAUDE.md", ROOT / "GEMINI.md"):
        text = path.read_text(encoding="utf-8")
        assert "README.md" in text
        assert "main.py" in text
        assert "src/tools" not in text
        assert "src/cases" not in text

    main_entry = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "load_runtime_config(cfg)" in main_entry
    assert "run_preflight(runtime_config)" in main_entry
    assert "run_case(runtime_config)" in main_entry

    run_case_module = (ROOT / "scripts/run_case.py").read_text(encoding="utf-8")
    assert "argparse" not in run_case_module
    assert "__main__" not in run_case_module
    assert 'user_instruction=str(runtime_config["experiment"]["user_instruction"])' in run_case_module

    preflight_module = (ROOT / "scripts/preflight.py").read_text(encoding="utf-8")
    assert "argparse" not in preflight_module
    assert "__main__" not in preflight_module


def test_root_config_is_baseline_only_and_requires_explicit_runtime_selection():
    root_config = (ROOT / "config/config.yaml").read_text(encoding="utf-8")
    assert "- optional data: null" in root_config
    assert "- optional experiment: null" in root_config
    assert "stage_b:" not in root_config
    assert "candidate_registry" not in root_config
    assert "dag_json" not in root_config
    assert "rm101_synth" not in root_config
    assert "dag_only" not in root_config

    cfg = compose_runtime_config()
    with pytest.raises(ValueError, match="must select a dataset"):
        to_runtime_dict(cfg)


def test_formal_run_presets_hold_runtime_semantics():
    expected = [
        ROOT / "config/runs/ottawa_ml.yaml",
        ROOT / "config/runs/ottawa_torch.yaml",
        ROOT / "config/runs/rm101_ml.yaml",
        ROOT / "config/runs/rm101_torch.yaml",
        ROOT / "config/runs/ottawa_ml_test.yaml",
        ROOT / "config/runs/rm101_ml_test.yaml",
    ]
    assert all(path.exists() for path in expected)

    for path in (
        ROOT / "config/runs/ottawa_ml.yaml",
        ROOT / "config/runs/ottawa_torch.yaml",
        ROOT / "config/runs/rm101_ml.yaml",
        ROOT / "config/runs/rm101_torch.yaml",
    ):
        text = path.read_text(encoding="utf-8")
        assert "provider: codex_cli" in text
        assert "mode: provider" in text
        assert "model: gpt-5.3-codex" in text
