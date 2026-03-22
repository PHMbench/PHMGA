from __future__ import annotations

from src.config import load_runtime_config


def test_load_runtime_config_resolves_layered_run_preset():
    config = load_runtime_config("rm101_ml_openrouter")

    assert config["data"]["dataset_name"] == "RM_101_THU_GEARBOX"
    assert config["experiment"]["graph_path"] == "ml"
    assert config["llm"]["provider"] == "openrouter"
    assert config["runtime"]["run_name"] == "rm101_ml_openrouter"


def test_load_runtime_config_accepts_legacy_case_alias():
    config = load_runtime_config("case_exp_ottawa")

    assert config["runtime"]["requested_run_name"] == "case_exp_ottawa"
    assert config["runtime"]["run_name"] == "ottawa_ml_gemini"
    assert config["llm"]["provider"] == "gemini"


def test_load_runtime_config_applies_cli_overrides():
    config = load_runtime_config(
        "rm101_torch_openrouter",
        action="preflight",
        output_dir="/tmp/phmga-test",
        overrides=["fusion.mode=fixed", "fusion.fixed_weights={a: 0.7, b: 0.3}"],
    )

    assert config["runtime"]["action"] == "preflight"
    assert config["runtime"]["output_dir"] == "/tmp/phmga-test"
    assert config["fusion"]["mode"] == "fixed"
    assert config["fusion"]["fixed_weights"] == {"a": 0.7, "b": 0.3}
