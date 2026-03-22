from __future__ import annotations

from pathlib import Path

from src.config import load_runtime_config
from src.runtime import run_experiment, run_preflight


def _apply_fixture(config: dict, fixture: dict[str, object], *, dataset_name: str, dataset_id: int) -> dict:
    config["data"]["metadata_path"] = fixture["metadata_path"]
    config["data"]["h5_path"] = fixture["h5_path"]
    config["data"]["selection"]["name"] = dataset_name
    config["data"]["selection"]["dataset_id"] = dataset_id
    config["data"]["split"] = {
        "strategy": "stratified_fixed_per_class",
        "seed": 0,
        "train_per_class": 2,
        "val_per_class": 1,
        "test_per_class": 1,
    }
    config["data"]["window"] = {
        "window_size": 64,
        "stride": 32,
        "slice_mode": "centered",
        "drop_last_window": False,
    }
    return config


def test_preflight_validates_rm101_preset(make_dataset_fixture, monkeypatch):
    fixture = make_dataset_fixture(dataset_name="RM_101_THU_GEARBOX", dataset_id=101)
    config = load_runtime_config("rm101_ml_openrouter")
    config = _apply_fixture(config, fixture, dataset_name="RM_101_THU_GEARBOX", dataset_id=101)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")

    payload = run_preflight(config)

    assert payload["status"] == "ok"
    assert payload["dataset_name"] == "RM_101_THU_GEARBOX"
    assert payload["llm"]["provider"] == "openrouter"


def test_offline_stub_smoke_run_emits_expected_artifacts(make_dataset_fixture, tmp_path: Path):
    fixture = make_dataset_fixture(dataset_name="RM_101_THU_GEARBOX", dataset_id=101)
    output_dir = tmp_path / "artifacts"
    config = load_runtime_config("rm101_ml_openrouter")
    config = _apply_fixture(config, fixture, dataset_name="RM_101_THU_GEARBOX", dataset_id=101)
    config["llm"]["mode"] = "offline_stub"
    config["runtime"]["output_dir"] = str(output_dir)

    payload = run_experiment(config)

    assert payload["status"] == "ok"
    for filename in (
        "resolved_config.json",
        "protocol.json",
        "dag.json",
        "feature_plan.json",
        "metrics.json",
        "branch_weights.json",
        "predictions.json",
        "final_report.md",
    ):
        assert (output_dir / filename).exists(), filename


def test_ottawa_gemini_provider_path_runs_with_monkeypatched_backend(make_dataset_fixture, tmp_path: Path, monkeypatch):
    fixture = make_dataset_fixture(dataset_name="RM_017_Ottawa19", dataset_id=13, channels=2)
    output_dir = tmp_path / "ottawa"
    config = load_runtime_config("ottawa_ml_gemini")
    config = _apply_fixture(config, fixture, dataset_name="RM_017_Ottawa19", dataset_id=13)
    config["runtime"]["output_dir"] = str(output_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")

    call_state = {"plan": 0, "reflect": 0}

    def fake_generate_json(self, prompt: str, *, repair_prompt: str | None = None):  # noqa: ARG001
        if "Current Stage" in prompt:
            call_state["reflect"] += 1
            if call_state["reflect"] == 1:
                return {"decision": "need_patch", "reason": "Build feature leaves first."}
            return {"decision": "finish", "reason": "Enough terminal features now."}
        call_state["plan"] += 1
        if call_state["plan"] == 1:
            return {"plan": [{"parent": "ch1", "op_name": "fft", "params": {}}, {"parent": "ch2", "op_name": "fft", "params": {}}]}
        return {
            "plan": [
                {"parent": "fft_01_ch1", "op_name": "mean", "params": {}},
                {"parent": "fft_02_ch2", "op_name": "mean", "params": {}},
            ]
        }

    monkeypatch.setattr("src.llm.backends.GeminiBackend.generate_json", fake_generate_json)
    monkeypatch.setattr("src.llm.backends.GeminiBackend.generate_text", lambda self, prompt: "# Gemini Report\n")

    payload = run_experiment(config)

    assert payload["status"] == "ok"
    assert (output_dir / "final_report.md").exists()
