from __future__ import annotations

import json
from pathlib import Path

import main as main_module
from src.cases.registry import register_case_runner


def test_main_runs_registered_case_via_hydra(tmp_path: Path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "graphs").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                "  - cases: demo",
                "  - graphs: builder_loop",
                "  - _self_",
                "name: fallback",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "cases" / "demo.yaml").write_text(
        "# @package _global_\ncases:\n  selected: structure_main_case\n",
        encoding="utf-8",
    )
    (tmp_path / "graphs" / "builder_loop.yaml").write_text(
        "selected: builder_loop\nbuilder_name: builder_loop\nexecutor_name: executor_tspn\n",
        encoding="utf-8",
    )

    observed: dict[str, str] = {}

    def _runner(runtime_config_path: str):
        observed["path"] = runtime_config_path
        payload = Path(runtime_config_path).read_text(encoding="utf-8")
        assert "structure_main_case" in payload

    register_case_runner("structure_main_case", _runner)

    rc = main_module.run_main(
        [
            "--config-dir",
            str(tmp_path),
            "--config-name",
            "config",
        ]
    )
    assert rc == 0
    assert Path(observed["path"]).exists()


def test_main_accepts_legacy_case_override(tmp_path: Path):
    config_path = tmp_path / "single.yaml"
    config_path.write_text("name: sample_case\n", encoding="utf-8")

    observed: dict[str, str] = {}

    def _runner(runtime_config_path: str):
        observed["path"] = runtime_config_path
        manifest = Path(runtime_config_path).with_name("runtime_manifest.json")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["selected_case"] == "structure_legacy_case"

    register_case_runner("structure_legacy_case", _runner)
    rc = main_module.run_main(
        [
            "structure_legacy_case",
            "--config",
            str(config_path),
        ]
    )
    assert rc == 0
    assert Path(observed["path"]).exists()
