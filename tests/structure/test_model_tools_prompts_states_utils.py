from __future__ import annotations

from pathlib import Path

import pytest

from src.model.bridge import DAG2ConfigAdapter
from src.prompts.shared import render_prompt_template
from src.states import BuilderState
from src.states.phm_states import DAGState, InputData
from src.tools.signal_processing_schemas import get_operator
from src.utils.paths import build_case_artifact_paths
from src.utils.serialization import write_json_file


def _make_builder_state() -> BuilderState:
    root = InputData(node_id="ch1", parents=[], shape=(1, 16, 1), data={}, results={"ref": {"a": []}, "tst": {"b": []}})
    dag = DAGState(user_instruction="demo", channels=["ch1"], nodes={"ch1": root}, leaves=["ch1"])
    return BuilderState(
        case_name="demo",
        user_instruction="demo",
        reference_signal=root,
        test_signal=root,
        dag_state=dag,
    )


def test_model_module_exposes_bridge_adapter():
    adapter = DAG2ConfigAdapter(in_dim=16, in_channels=1, num_classes=2)
    assert adapter.in_dim == 16


def test_tools_module_keeps_operator_registry():
    op = get_operator("mean")
    assert op.op_name == "mean"


def test_prompts_module_renders_without_env_access():
    rendered = render_prompt_template("hello {name}", name="phmga")
    assert rendered == "hello phmga"


def test_states_module_supports_builder_state_round_trip():
    state = _make_builder_state()
    payload = state.model_dump()
    assert payload["case_name"] == "demo"
    assert payload["dag_state"]["channels"] == ["ch1"]


def test_utils_module_writes_json_and_paths(tmp_path: Path):
    paths = build_case_artifact_paths("demo", tmp_path)
    out_path = write_json_file({"ok": True}, Path(paths["case_dir"]) / "meta.json")
    assert out_path.exists()
    assert Path(paths["report_path"]).name == "final_report.md"


def test_schemas_module_rejects_invalid_data_selection():
    from src.schemas.config_schema import DataSelectionSpec

    with pytest.raises(Exception):
        DataSelectionSpec(mode="invalid")
