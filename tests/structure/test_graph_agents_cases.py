from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.agents.builder.plan import plan_agent as builder_plan_agent
from src.agents.plan_agent import plan_agent as legacy_plan_agent
from src.cases.base_runner import prepare_case_runtime, run_registered_case
from src.cases.registry import get_case_runner, list_case_runners, register_case_runner
from src.graph import build_builder_graph, list_graph_builders, register_graph_builder
from src.graph.registry import build_selected_graph
from src.phm_outer_graph import build_builder_graph as legacy_build_builder_graph


def test_graph_module_registry_selects_registered_builder():
    register_graph_builder("structure_fake_graph", lambda: {"name": "fake"})
    assert "structure_fake_graph" in list_graph_builders()
    assert build_selected_graph("structure_fake_graph") == {"name": "fake"}
    assert callable(build_builder_graph)
    assert callable(legacy_build_builder_graph)


def test_agents_module_wrapper_forwards_old_entrypoint():
    assert builder_plan_agent is legacy_plan_agent


def test_cases_module_prepares_runtime_artifacts(tmp_path: Path):
    (tmp_path / "system").mkdir()
    (tmp_path / "graphs").mkdir()
    (tmp_path / "cases").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                "  - system/runtime",
                "  - graphs: builder_loop",
                "  - cases: demo",
                "  - _self_",
                "project:",
                "  name: case_runtime",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "system" / "runtime.yaml").write_text("runtime:\n  profile: smoke\n", encoding="utf-8")
    (tmp_path / "graphs" / "builder_loop.yaml").write_text("selected: builder_loop\n", encoding="utf-8")
    (tmp_path / "cases" / "demo.yaml").write_text(
        "# @package _global_\ncases:\n  selected: demo_case\nname: demo_case\n",
        encoding="utf-8",
    )

    paths = prepare_case_runtime(
        config_path=config_path,
        save_root=tmp_path / "save",
        metadata_snapshot={"row_count": 1, "columns": ["Id"]},
    )
    resolved = json.loads(Path(paths["resolved_config_path"]).read_text(encoding="utf-8"))
    metadata = json.loads(Path(paths["metadata_snapshot_path"]).read_text(encoding="utf-8"))
    assert resolved["project"]["name"] == "case_runtime"
    assert metadata["row_count"] == 1
    assert paths["selected_case"] == "demo_case"
    assert Path(paths["runtime_config_path"]).exists()


def test_cases_registry_registers_fake_runner():
    register_case_runner("structure_fake_case", lambda *_args, **_kwargs: "ok")
    assert get_case_runner("structure_fake_case")() == "ok"
    assert "case1" in list_case_runners()


def test_cases_runner_executes_registered_case(tmp_path: Path):
    (tmp_path / "cases").mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "defaults:",
                "  - cases: demo",
                "  - _self_",
                "name: fallback_name",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "cases" / "demo.yaml").write_text(
        "# @package _global_\ncases:\n  selected: structure_runner_case\n",
        encoding="utf-8",
    )

    observed = {}

    def _runner(runtime_config_path: str):
        observed["path"] = runtime_config_path

    register_case_runner("structure_runner_case", _runner)
    runtime = run_registered_case(config_path=config_path, save_root=tmp_path / "save")
    assert runtime["selected_case"] == "structure_runner_case"
    assert Path(observed["path"]).exists()


def test_phm_outer_graph_is_thin_facade():
    import src.phm_outer_graph as facade

    source = inspect.getsource(facade)
    assert "src.graph" in source
    assert "StateGraph(" not in source
