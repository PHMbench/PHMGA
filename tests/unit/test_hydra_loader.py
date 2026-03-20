from __future__ import annotations

from pathlib import Path

from src.config import compose_runtime_config, load_runtime_config, to_runtime_dict


ROOT = Path(__file__).resolve().parents[2]


def test_load_runtime_config_still_accepts_run_preset_paths():
    runtime_config = load_runtime_config(ROOT / "config/runs/rm101_synth_dag.yaml")

    assert runtime_config["data"]["dataset_name"] == "RM101_SYNTH"
    assert runtime_config["experiment"]["graph_path"] == "dag_only"
    assert runtime_config["experiment"]["user_instruction"] == "Generate a paper-ready PHM workflow from canonical metadata."
    assert runtime_config["runtime"]["config_name"] == "rm101_synth_dag"
    assert runtime_config["runtime"]["action"] == "run_case"


def test_hydra_compose_and_plain_dict_conversion_match_path_loading():
    cfg = compose_runtime_config(overrides=["+runs=rm101_synth_ml"])
    runtime_config = to_runtime_dict(cfg)

    assert runtime_config["data"]["dataset_name"] == "RM101_SYNTH"
    assert runtime_config["experiment"]["graph_path"] == "ml"
    assert runtime_config["experiment"]["user_instruction"] == "Generate a paper-ready PHM workflow from canonical metadata."
    assert runtime_config["runtime"]["action"] == "run_case"
    assert runtime_config["runtime"]["workflow_mode"] == "rich"
    assert runtime_config["runtime"]["config_path"] == "<hydra>"


def test_proving_run_preset_sets_supervisor_workflow_mode():
    cfg = compose_runtime_config(overrides=["+runs=ottawa_ml_codex_proving", "data=ottawa_synth"])
    runtime_config = to_runtime_dict(cfg)

    assert runtime_config["data"]["dataset_name"] == "OTTAWA_SYNTH"
    assert runtime_config["experiment"]["graph_path"] == "ml"
    assert runtime_config["runtime"]["workflow_mode"] == "supervisor_proving"
    assert runtime_config["runtime"]["max_iterations"] == 1
    assert runtime_config["evaluation"]["dag_quality"]["enabled"] is False
