from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phm_core import DAGState, InputData, PHMState
from src.cases.case1 import _bind_llm_from_case, run_case
from src.config import normalize_runtime_config, resolve_data_selection
from src.agents.deep_model_train_agent import _resolve_tspn_config
from src.utils.preflight import build_preflight_report


def test_bind_llm_from_case_overrides_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("QUERY_GENERATOR_MODEL", "openai/gpt-4o-mini")
    cfg = {
        "llm": {
            "provider": "openrouter",
            "query_generator_model": "openai/gpt-4.1-mini",
        }
    }

    source = _bind_llm_from_case(cfg)
    assert source == "case_yaml"
    assert os.getenv("LLM_PROVIDER") == "openrouter"
    assert os.getenv("QUERY_GENERATOR_MODEL") == "openai/gpt-4.1-mini"
    assert os.getenv("PHM_MODEL") == "openai/gpt-4.1-mini"
    assert cfg["llm"]["provider"] == "openrouter"


def test_bind_llm_from_case_rejects_invalid_provider():
    cfg = {"llm": {"provider": "bad-provider", "query_generator_model": "x"}}
    with pytest.raises(ValueError, match="Invalid llm.provider"):
        _bind_llm_from_case(cfg)


def test_preflight_detects_provider_model_mismatch():
    cfg = {
        "data": {"source_mode": "fixed_ids", "selection": {"mode": "fixed_ids", "train_ids": [1], "test_ids": [2]}},
        "metadata_path": __file__,
        "h5_path": __file__,
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "0",
            "LLM_PROVIDER": "glm",
            "QUERY_GENERATOR_MODEL": "openai/gpt-4o-mini",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY": "dummy",
        },
    )
    assert report["ok"] is False
    assert any("Invalid llm.provider" in item for item in report["errors"])
    assert "provider_checks" in report["checks"]
    assert report["checks"]["provider_source"] == "env"


def test_preflight_fake_llm_downgrades_provider_errors_to_warnings():
    cfg = {
        "data": {"source_mode": "fixed_ids", "selection": {"mode": "fixed_ids", "train_ids": [1], "test_ids": [2]}},
        "metadata_path": __file__,
        "h5_path": __file__,
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "true",
            "LLM_PROVIDER": "openrouter",
            "QUERY_GENERATOR_MODEL": "openai/gpt-4o-mini",
            "OPENROUTER_BASE_URL": "",
            "OPENROUTER_API_KEY": "",
        },
    )
    assert report["ok"] is True
    assert not report["errors"]
    assert any("[FAKE_LLM]" in item for item in report["warnings"])


def test_preflight_uses_case_llm_over_env():
    cfg = {
        "data": {"source_mode": "fixed_ids", "selection": {"mode": "fixed_ids", "train_ids": [1], "test_ids": [2]}},
        "metadata_path": __file__,
        "h5_path": __file__,
        "llm": {
            "provider": "openrouter",
            "query_generator_model": "openai/gpt-4o-mini",
        },
    }
    report = build_preflight_report(
        cfg,
        env={
            "FAKE_LLM": "0",
            "LLM_PROVIDER": "glm",
            "QUERY_GENERATOR_MODEL": "GLM-4.7-Flash",
            "OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENROUTER_API_KEY": "dummy",
        },
    )
    assert report["checks"]["provider_source"] == "case_yaml"
    assert report["checks"]["provider_checks"]["provider"] == "openrouter"
    assert report["ok"] is True


def test_runtime_config_normalizes_legacy_fixed_ids_to_data_selection():
    cfg = {
        "metadata_path": __file__,
        "h5_path": __file__,
        "ref_ids": [11, 12],
        "test_ids": [21],
        "data": {"source_mode": "fixed_ids"},
    }
    normalized = normalize_runtime_config(cfg)
    selection = resolve_data_selection(normalized)
    assert selection.train_ids == [11, 12]
    assert selection.val_ids == []
    assert selection.test_ids == [21]
    assert "ref_ids" not in normalized
    assert "test_ids" not in normalized
    assert "ref_ids" not in normalized["data"]
    assert "test_ids" not in normalized["data"]


def test_case1_loaded_state_receives_resolved_model_truth(monkeypatch, tmp_path):
    save_root = tmp_path / "save"
    state_path = tmp_path / "built_state.pkl"
    state_path.write_bytes(b"stub")
    report_path = tmp_path / "final_report.md"
    config_path = tmp_path / "case.yaml"

    cfg = {
        "name": "case1_model_truth",
        "save_dir": str(save_root),
        "state_save_path": str(state_path),
        "report_path": str(report_path),
        "user_instruction": "diagnose",
        "run_executor": False,
        "train_backend": "tspn",
        "metadata_path": __file__,
        "h5_path": __file__,
        "data": {
            "source_mode": "fixed_ids",
            "selection": {"mode": "fixed_ids", "train_ids": [1], "test_ids": [2]},
        },
        "model": {
            "config_path": "config/model_tspn_basic.yaml",
            "autofit_dims": False,
            "autofit_num_classes": False,
        },
        "preflight": {"strict": True},
    }
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    sig = np.zeros((1, 32, 1), dtype=np.float32)
    ch1 = InputData(node_id="ch1", data={"signal": sig}, results={"train": {"s0": sig}}, parents=[], shape=sig.shape, meta={})
    dag = DAGState(user_instruction="case", channels=["ch1"], nodes={"ch1": ch1}, leaves=["ch1"])
    loaded_state = PHMState(
        user_instruction="case",
        reference_signal=ch1,
        test_signal=ch1,
        dag_state=dag,
        case_name="case1_model_truth",
        save_dir=str(save_root),
        train_backend="tspn",
    )

    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.setattr("src.cases.case1.load_state", lambda _path: loaded_state)
    run_case(str(config_path))

    assert loaded_state.model_config_path == "config/model_tspn_basic.yaml"
    assert loaded_state.model_cfg == cfg["model"]


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
        "openrouter",
        "--model",
        "openai/gpt-4o-mini",
        "--compat-profile",
        "rm101_strict",
    ]
    subprocess.check_call(cmd, cwd=str(ROOT))

    cfg = yaml.safe_load(out_cfg.read_text(encoding="utf-8")) or {}
    assert cfg["llm"]["provider"] == "openrouter"
    assert cfg["llm"]["query_generator_model"] == "openai/gpt-4o-mini"
    assert cfg["llm"]["phm_model"] == "openai/gpt-4o-mini"
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
