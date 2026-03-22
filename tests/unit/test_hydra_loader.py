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


def test_simple_fullchain_run_preset_sets_simple_workflow_mode():
    cfg = compose_runtime_config(overrides=["+runs=ottawa_synth_ml_simple"])
    runtime_config = to_runtime_dict(cfg)

    assert runtime_config["data"]["dataset_name"] == "OTTAWA_SYNTH"
    assert runtime_config["experiment"]["graph_path"] == "ml"
    assert runtime_config["runtime"]["workflow_mode"] == "simple_fullchain"
    assert runtime_config["runtime"]["max_iterations"] == 2
    assert runtime_config["evaluation"]["dag_quality"]["enabled"] is False


def test_real_simple_run_presets_freeze_real_smoke_contract():
    for preset_name, dataset_name, output_suffix in (
        ("ottawa_ml_codex_simple", "RM_017_Ottawa19", "artifacts/simple/ottawa_ml_codex_simple"),
        ("rm101_ml_codex_simple", "RM_101_THU_GEARBOX", "artifacts/simple/rm101_ml_codex_simple"),
    ):
        cfg = compose_runtime_config(overrides=[f"+runs={preset_name}"])
        runtime_config = to_runtime_dict(cfg)

        assert runtime_config["data"]["dataset_name"] == dataset_name
        assert runtime_config["experiment"]["graph_path"] == "ml"
        assert runtime_config["llm"]["provider"] == "codex_cli"
        assert runtime_config["llm"]["model"] == "gpt-5.3-codex"
        assert runtime_config["runtime"]["workflow_mode"] == "simple_fullchain"
        assert runtime_config["runtime"]["max_iterations"] == 3
        assert runtime_config["runtime"]["min_depth"] == 2
        assert runtime_config["runtime"]["min_width"] == 1
        assert runtime_config["runtime"]["max_depth"] == 8
        assert runtime_config["runtime"]["output_dir"].endswith(output_suffix)
        assert runtime_config["evaluation"]["dag_quality"]["enabled"] is False
        assert runtime_config["evaluation"]["dag_quality"]["use_proxy_probe"] is False
        assert runtime_config["model"]["ml"]["max_iter"] == 50
        assert runtime_config["data"]["split"]["strategy"] == "stratified_fixed_per_class"
        assert runtime_config["data"]["split"]["train_per_class"] == 2
        assert runtime_config["data"]["split"]["val_per_class"] == 1
        assert runtime_config["data"]["split"]["test_per_class"] == 1
        assert runtime_config["data"]["window"]["slice_mode"] == "centered"


def test_formal_v3_run_preset_freezes_depth_constraints_and_output_dir():
    cfg = compose_runtime_config(overrides=["+runs=ottawa_ml_openrouter_nemotron_v3"])
    runtime_config = to_runtime_dict(cfg)

    assert runtime_config["data"]["dataset_name"] == "RM_017_Ottawa19"
    assert runtime_config["experiment"]["graph_path"] == "ml"
    assert runtime_config["llm"]["provider"] == "openrouter"
    assert runtime_config["llm"]["model"] == "nvidia/nemotron-3-super-120b-a12b:free"
    assert runtime_config["runtime"]["workflow_mode"] == "rich"
    assert runtime_config["runtime"]["min_depth"] == 3
    assert runtime_config["runtime"]["min_width"] == 1
    assert runtime_config["runtime"]["max_depth"] == 8
    assert runtime_config["runtime"]["output_dir"].endswith("artifacts/paper/ottawa_ml_openrouter_nemotron_v3")
