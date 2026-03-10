from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cases.case1 import _bind_llm_from_case
from src.agents.deep_model_train_agent import _resolve_tspn_config
from src.utils.preflight import build_preflight_report


def test_bind_llm_from_case_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL", "gemini-2.5-pro")
    cfg = {
        "llm": {
            "provider": "glm",
            "query_generator_model": "GLM-4.7-Flash",
        }
    }

    source = _bind_llm_from_case(cfg)
    assert source == "case_yaml"
    assert os.getenv("LLM_PROVIDER") == "glm"
    assert os.getenv("QUERY_GENERATOR_MODEL") == "GLM-4.7-Flash"
    assert os.getenv("PHM_MODEL") == "GLM-4.7-Flash"
    assert cfg["llm"]["provider"] == "glm"


def test_bind_llm_from_case_rejects_invalid_provider():
    cfg = {"llm": {"provider": "bad-provider", "query_generator_model": "x"}}
    with pytest.raises(ValueError, match="Invalid llm.provider"):
        _bind_llm_from_case(cfg)


def test_preflight_detects_provider_model_mismatch():
    cfg = {
        "data": {"source_mode": "fixed_ids"},
        "metadata_path": __file__,
        "h5_path": __file__,
        "ref_ids": [1],
        "test_ids": [2],
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "0",
            "LLM_PROVIDER": "glm",
            "QUERY_GENERATOR_MODEL": "gemini-2.5-pro",
            "GLM_API_BASE": "https://open.bigmodel.cn/api/paas/v4",
            "GLM_API_KEY": "dummy",
        },
    )
    assert report["ok"] is False
    assert any("Gemini" in item for item in report["errors"])
    assert "provider_checks" in report["checks"]
    assert report["checks"]["provider_source"] == "env"


def test_preflight_fake_llm_downgrades_provider_errors_to_warnings():
    cfg = {
        "data": {"source_mode": "fixed_ids"},
        "metadata_path": __file__,
        "h5_path": __file__,
        "ref_ids": [1],
        "test_ids": [2],
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "true",
            "LLM_PROVIDER": "glm",
            "QUERY_GENERATOR_MODEL": "GLM-4.7-Flash",
            "GLM_API_BASE": "",
            "GLM_API_KEY": "",
        },
    )
    assert report["ok"] is True
    assert not report["errors"]
    assert any("[FAKE_LLM]" in item for item in report["warnings"])


def test_preflight_uses_case_llm_over_env():
    cfg = {
        "data": {"source_mode": "fixed_ids"},
        "metadata_path": __file__,
        "h5_path": __file__,
        "ref_ids": [1],
        "test_ids": [2],
        "llm": {
            "provider": "glm",
            "query_generator_model": "GLM-4.7-Flash",
        },
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "0",
            "LLM_PROVIDER": "gemini",
            "QUERY_GENERATOR_MODEL": "gemini-2.5-pro",
            "GLM_API_BASE": "https://open.bigmodel.cn/api/paas/v4",
            "GLM_API_KEY": "dummy",
        },
    )
    assert report["checks"]["provider_source"] == "case_yaml"
    assert report["checks"]["provider_checks"]["provider"] == "glm"
    assert report["ok"] is True


def test_resolve_tspn_config_num_classes_fail_fast_and_autofit():
    source = str(Path("config") / "model_tspn_basic.yaml")

    with pytest.raises(ValueError, match="num_classes mismatch"):
        _resolve_tspn_config(
            source_model_config_path=source,
            inferred_in_dim=4096,
            inferred_in_channels=2,
            inferred_num_classes=3,
            autofit_dims=True,
            autofit_num_classes=False,
        )

    resolved, info = _resolve_tspn_config(
        source_model_config_path=source,
        inferred_in_dim=2048,
        inferred_in_channels=1,
        inferred_num_classes=3,
        autofit_dims=True,
        autofit_num_classes=True,
    )
    assert resolved.model.in_dim == 2048
    assert resolved.model.in_channels == 1
    assert resolved.model.num_classes == 3
    assert "overrides" in info


def test_resolve_case_config_writes_llm_block(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text("name: base\nstate_save_path: a\nreport_path: b\n", encoding="utf-8")
    out_cfg = tmp_path / "resolved.yaml"

    cmd = [
        sys.executable,
        "scripts/paper/resolve_case_config.py",
        "--base-config",
        str(base_cfg),
        "--out-config",
        str(out_cfg),
        "--case-name",
        "paper_test",
        "--save-root",
        str(tmp_path / "save"),
        "--ablation-mode",
        "full",
        "--train-backend",
        "tspn",
        "--provider",
        "glm",
        "--model",
        "GLM-4.7-Flash",
        "--compat-profile",
        "rm101_strict",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))

    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8")) or {}
    assert cfg["llm"]["provider"] == "glm"
    assert cfg["llm"]["query_generator_model"] == "GLM-4.7-Flash"
    assert cfg["llm"]["phm_model"] == "GLM-4.7-Flash"
    assert cfg["data"]["compat_profile"] == "rm101_strict"
    assert cfg["data"]["operator_contract"] == "rm101_closed_v1"
    assert cfg["data"]["enforce_tspn_closed_world"] is True
    assert cfg["data"]["state_save_mode"] == "auto"


def test_resolve_case_config_writes_state_save_mode(tmp_path):
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text("name: base\nstate_save_path: a\nreport_path: b\n", encoding="utf-8")
    out_cfg = tmp_path / "resolved.yaml"

    cmd = [
        sys.executable,
        "scripts/paper/resolve_case_config.py",
        "--base-config",
        str(base_cfg),
        "--out-config",
        str(out_cfg),
        "--case-name",
        "paper_test_state_mode",
        "--save-root",
        str(tmp_path / "save"),
        "--ablation-mode",
        "full",
        "--train-backend",
        "tspn",
        "--state-save-mode",
        "minimal",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))

    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8")) or {}
    assert cfg["data"]["state_save_mode"] == "minimal"


def test_discover_run_artifacts_generates_fallback_metrics(tmp_path):
    case_dir = tmp_path / "paper_case"
    (case_dir / "run-100" / "logs").mkdir(parents=True)
    (case_dir / "run-100" / "logs" / "run.log").write_text("", encoding="utf-8")
    (case_dir / "final_report.md").write_text("# report\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/paper/discover_run_artifacts.py",
        "--case-dir",
        str(case_dir),
    ]
    raw = subprocess.check_output(cmd, cwd=str(ROOT), text=True).strip()
    payload = json.loads(raw)

    assert payload["metrics_path"]
    metrics_path = Path(payload["metrics_path"])
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("is_fallback") is True
