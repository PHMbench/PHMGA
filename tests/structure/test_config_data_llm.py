from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    bind_llm_env,
    load_composed_config,
    normalize_runtime_config,
    normalize_llm_config,
    resolve_config,
    resolve_data_selection,
    resolve_source_mode,
    select_fixed_ids,
    validate_llm_config,
    validate_metadata_columns,
    write_resolved_config,
)


def test_config_module_resolves_minimal_composed_config(tmp_path: Path):
    (tmp_path / "system").mkdir()
    (tmp_path / "graphs").mkdir()
    root = tmp_path / "config.yaml"
    root.write_text(
        "\n".join(
            [
                "defaults:",
                "  - system/runtime",
                "  - graphs/executor_tspn",
                "  - _self_",
                "project:",
                "  name: test_project",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "system" / "runtime.yaml").write_text("runtime:\n  profile: smoke\n", encoding="utf-8")
    (tmp_path / "graphs" / "executor_tspn.yaml").write_text("selected: executor_tspn\n", encoding="utf-8")

    resolved = resolve_config(root, overrides={"project.version": "test"})
    assert resolved.project["name"] == "test_project"
    assert resolved.project["version"] == "test"
    assert resolved.graphs["selected"] == "executor_tspn"
    out = write_resolved_config(resolved, tmp_path / "resolved.json")
    assert json.loads(out.read_text(encoding="utf-8"))["project"]["name"] == "test_project"


def test_data_module_validates_metadata_and_selects_fixed_ids():
    frame = pd.DataFrame(
        [
            {"Id": 1, "Dataset_id": "d1", "Name": "a", "Type": "sig", "File": "a.npy", "Label": 0, "Label_Description": "ok", "Sample_Rate": 1, "Length": 32, "Channels": 1},
            {"Id": 2, "Dataset_id": "d1", "Name": "b", "Type": "sig", "File": "b.npy", "Label": 1, "Label_Description": "fault", "Sample_Rate": 1, "Length": 32, "Channels": 1},
            {"Id": 3, "Dataset_id": "d1", "Name": "c", "Type": "sig", "File": "c.npy", "Label": 1, "Label_Description": "fault", "Sample_Rate": 1, "Length": 32, "Channels": 1},
        ]
    )
    report = validate_metadata_columns(frame)
    assert report["ok"] is True
    selected = select_fixed_ids(frame, train_ids=[1, 2], val_ids=[3], test_ids=[2])
    assert list(selected["train"]["Id"]) == [1, 2]
    assert list(selected["val"]["Id"]) == [3]
    assert list(selected["test"]["Id"]) == [2]
    assert resolve_source_mode({"data": {"backend": "vibench"}}) == "vibench"


def test_data_module_normalizes_legacy_fixed_id_keys_into_canonical_selection():
    normalized = normalize_runtime_config(
        {
            "data": {"source_mode": "fixed_ids"},
            "ref_ids": [10, 11],
            "val_ids": [12],
            "test_ids": [13],
        }
    )
    selection = resolve_data_selection(normalized)
    assert selection.train_ids == [10, 11]
    assert selection.val_ids == [12]
    assert selection.test_ids == [13]
    assert "ref_ids" not in normalized
    assert "test_ids" not in normalized


def test_llm_module_normalizes_and_validates_without_network():
    llm_cfg = {"provider": "openrouter", "query_generator_model": "openai/gpt-4o-mini"}
    env = {"OPENROUTER_BASE_URL": "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY": "dummy"}
    normalized = normalize_llm_config(llm_cfg, env=env)
    assert normalized["provider"] == "openrouter"
    assert normalized["query_generator_model"] == "openai/gpt-4o-mini"
    bound = bind_llm_env(llm_cfg, env=env)
    assert bound["PHM_MODEL"] == "openai/gpt-4o-mini"
    report = validate_llm_config(llm_cfg, env=env)
    assert report["ok"] is True
    assert report["source"] == "case_yaml"


def test_config_module_supports_hydra_group_override(tmp_path: Path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "graphs").mkdir()
    root = tmp_path / "config.yaml"
    root.write_text(
        "\n".join(
            [
                "defaults:",
                "  - cases: default",
                "  - graphs: builder",
                "  - _self_",
                "project:",
                "  name: hydra_project",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "cases" / "default.yaml").write_text("# @package _global_\ncases:\n  selected: default_case\n", encoding="utf-8")
    (tmp_path / "cases" / "alt.yaml").write_text("# @package _global_\ncases:\n  selected: alt_case\n", encoding="utf-8")
    (tmp_path / "graphs" / "builder.yaml").write_text("selected: builder_loop\n", encoding="utf-8")

    payload, _sources = load_composed_config(root, overrides=["cases=alt"])
    assert payload["cases"]["selected"] == "alt_case"


def test_config_module_invalid_hydra_override_fails(tmp_path: Path):
    root = tmp_path / "config.yaml"
    root.write_text("project:\n  name: hydra_project\n", encoding="utf-8")
    with pytest.raises(Exception):
        load_composed_config(root, overrides=["missing_group=bad"])


def test_schema_validation_rejects_invalid_graph_selection():
    from src.schemas.config_schema import GraphSelectionSpec

    with pytest.raises(Exception):
        GraphSelectionSpec(selected="")
