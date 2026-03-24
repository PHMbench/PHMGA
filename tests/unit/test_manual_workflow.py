from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.manual_workflow import export_dag_artifacts, load_node_datasets, run_manual_postprocess
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def _build_state(tmp_path: Path) -> PHMState:
    ref_features = tmp_path / "ref_features.npz"
    tst_features = tmp_path / "tst_features.npz"
    np.savez(
        ref_features,
        ref_a=np.array([1.0, 0.0]),
        ref_b=np.array([0.0, 1.0]),
    )
    np.savez(
        tst_features,
        test_a=np.array([0.9, 0.1]),
        test_b=np.array([0.2, 0.8]),
    )

    root = InputData(
        node_id="ch1",
        parents=[],
        shape=(2,),
        data={},
        results={
            "ref": {"ref_a": np.array([1.0, 0.0]), "ref_b": np.array([0.0, 1.0])},
            "tst": {"test_a": np.array([0.9, 0.1]), "test_b": np.array([0.2, 0.8])},
        },
        metadata={},
        meta={
            "labels": {"ref_a": 0, "ref_b": 1, "test_a": 0, "test_b": 1},
            "channel": "ch1",
        },
    )
    processed = ProcessedData(
        node_id="fft_01_ch1",
        parents=["ch1"],
        source_signal_id="ch1",
        method="fft",
        results={"ref": None, "tst": None},
        meta={
            "channel": "ch1",
            "saved": {"ref_path": str(ref_features), "tst_path": str(tst_features)},
        },
        shape=(2,),
    )
    dag = DAGState(
        user_instruction="diagnose bearing faults",
        channels=["ch1"],
        nodes={"ch1": root, "fft_01_ch1": processed},
        leaves=["fft_01_ch1"],
    )
    return PHMState(
        case_name="demo_case",
        user_instruction="diagnose bearing faults",
        reference_signal=root,
        test_signal=root,
        dag_state=dag,
        runtime_config={"runtime": {"output_dir": str(tmp_path / "artifacts")}},
    )


def test_export_dag_artifacts_writes_json_and_png(tmp_path: Path):
    state = _build_state(tmp_path)

    artifacts = export_dag_artifacts(state, output_dir=tmp_path / "graphs", stem="dag", max_nodes=None)

    assert artifacts["warnings"] == []
    json_path = Path(artifacts["json_path"])
    png_path = Path(artifacts["png_path"])
    assert json_path.exists()
    assert png_path.exists()
    assert png_path.stat().st_size > 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    node_ids = {item["node_id"] for item in payload["graph"]}
    assert {"ch1", "fft_01_ch1"}.issubset(node_ids)


def test_run_manual_postprocess_supports_function_level_reuse(tmp_path: Path, monkeypatch):
    state = _build_state(tmp_path)

    monkeypatch.setattr(
        "src.manual_workflow.report_agent_node",
        lambda state: {"final_report": "# final report\nmanual path"},
    )

    result = run_manual_postprocess(
        state,
        dataset_output_dir=tmp_path / "datasets",
        ml_results_path=tmp_path / "ml_results.pkl",
        metrics_markdown_path=tmp_path / "metrics.md",
        report_path=tmp_path / "final_report.md",
        cv_folds=2,
    )

    assert result["n_nodes"] == 1
    assert result["final_report"].startswith("# final report")
    assert result["ml_results"]["models"]

    dataset_dir = tmp_path / "datasets"
    saved = load_node_datasets(dataset_dir)
    assert "fft_01_ch1" in saved
    assert saved["fft_01_ch1"]["X_train"].shape[0] == 2

    assert (tmp_path / "ml_results.pkl").exists()
    assert (tmp_path / "metrics.md").exists()
    assert (tmp_path / "final_report.md").exists()
