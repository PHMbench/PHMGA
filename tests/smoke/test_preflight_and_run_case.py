from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
REAL_RM101_METADATA = Path("/home/user/data/PHMbenchdata/PHM-Vibench/gear_metadata.xlsx")
REAL_RM101_H5 = Path("/home/user/data/PHMbenchdata/PHM-Vibench/RM_101_THU_GEARBOX.h5")
REAL_OTTAWA_METADATA = Path("/home/user/data/PHMbenchdata/PHM-Vibench/metadata.xlsx")
REAL_OTTAWA_H5 = Path("/home/user/data/PHMbenchdata/PHM-Vibench/RM_017_Ottawa19.h5")


def test_preflight_reports_one_resolved_config():
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", "config/runs/rm101_synth_dag.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["dataset_name"] == "RM101_SYNTH"
    assert payload["graph_path"] == "dag_only"
    assert payload["source_mode"] == "synthetic"


def test_run_case_all_synthetic_path_pairs(tmp_path: Path):
    combos = [
        ("config/runs/rm101_synth_dag.yaml", "dag_only", "dag_artifacts.json"),
        ("config/runs/rm101_synth_ml.yaml", "ml", "feature_pipeline.json"),
        ("config/runs/rm101_synth_torch.yaml", "torch", "model_build_plan.json"),
        ("config/runs/ottawa_synth_dag.yaml", "dag_only", "dag_artifacts.json"),
        ("config/runs/ottawa_synth_ml.yaml", "ml", "feature_pipeline.json"),
        ("config/runs/ottawa_synth_torch.yaml", "torch", "model_build_plan.json"),
    ]
    for config_name, graph_path, expected_file in combos:
        output_dir = tmp_path / Path(config_name).stem
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_case.py",
                "--config",
                config_name,
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["graph_path"] == graph_path
        assert (output_dir / "dag.json").exists()
        assert (output_dir / "validated_dag.json").exists()
        assert (output_dir / "compiled_dag_manifest.json").exists()
        assert (output_dir / "resolved_splits.json").exists()
        assert (output_dir / "resolved_dataset_manifest.json").exists()
        assert (output_dir / "artifact_index.json").exists()
        assert (output_dir / "workflow_state.json").exists()
        assert (output_dir / "dag_quality_summary.json").exists()
        assert (output_dir / "decision_side_outputs.json").exists()
        assert (output_dir / "final_report.md").exists()
        assert (output_dir / expected_file).exists()
        if graph_path in {"ml", "torch"}:
            assert (output_dir / "similarity_artifacts.json").exists()
        if graph_path == "ml":
            assert (output_dir / "feature_list.json").exists()
            assert (output_dir / "feature_separability_summary.json").exists()
            assert (output_dir / "dataset_level_runtime_trace.json").exists()
            summary = json.loads((output_dir / "feature_separability_summary.json").read_text(encoding="utf-8"))
            assert summary["graph_path"] == "ml"
            assert "top_features" in summary
            assert "aggregate_scores" in summary
            assert "split_stability" in summary
            runtime_trace = json.loads((output_dir / "dataset_level_runtime_trace.json").read_text(encoding="utf-8"))
            assert runtime_trace["graph_path"] == "ml"
            assert "splits" in runtime_trace
            artifact_index = json.loads((output_dir / "artifact_index.json").read_text(encoding="utf-8"))
            assert "dataset_level_runtime_trace.json" in artifact_index


@pytest.mark.skipif(
    not all(path.exists() for path in (REAL_RM101_METADATA, REAL_RM101_H5, REAL_OTTAWA_METADATA, REAL_OTTAWA_H5)),
    reason="Real PHM-Vibench files are not available in this environment.",
)
def test_real_configs_preflight_and_dag_only(tmp_path: Path):
    for config_name, expected_dataset in (
        ("config/runs/rm101_dag_test.yaml", "RM_101_THU_GEARBOX"),
        ("config/runs/ottawa_dag_test.yaml", "RM_017_Ottawa19"),
    ):
        preflight = subprocess.run(
            [sys.executable, "scripts/preflight.py", "--config", config_name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        preflight_payload = json.loads(preflight.stdout)
        assert preflight_payload["dataset_name"] == expected_dataset
        assert preflight_payload["source_mode"] == "real"

        output_dir = tmp_path / f"{Path(config_name).stem}_real"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_case.py",
                "--config",
                config_name,
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["dataset"] == expected_dataset
        assert payload["graph_path"] == "dag_only"
        assert payload["source_mode"] == "real"
        assert (output_dir / "dag_artifacts.json").exists()
        assert (output_dir / "validated_dag.json").exists()
        assert (output_dir / "artifact_index.json").exists()
        assert (output_dir / "decision_side_outputs.json").exists()
        assert (output_dir / "resolved_splits.json").exists()
        assert (output_dir / "resolved_dataset_manifest.json").exists()
        assert (output_dir / "dag_quality_summary.json").exists()


@pytest.mark.skipif(
    not all(path.exists() for path in (REAL_OTTAWA_METADATA, REAL_OTTAWA_H5)),
    reason="Real Ottawa PHM-Vibench files are not available in this environment.",
)
@pytest.mark.parametrize(
    ("config_name", "graph_path", "expected_file", "report_section"),
    (
        ("config/runs/ottawa_ml_test.yaml", "ml", "feature_pipeline.json", "## ML Evidence"),
        ("config/runs/ottawa_torch_test.yaml", "torch", "model_build_plan.json", "## Torch Evidence"),
    ),
)
def test_real_ottawa_ml_and_torch_smoke(tmp_path: Path, config_name: str, graph_path: str, expected_file: str, report_section: str):
    preflight = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", config_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    preflight_payload = json.loads(preflight.stdout)
    assert preflight_payload["dataset_name"] == "RM_017_Ottawa19"
    assert preflight_payload["graph_path"] == graph_path
    assert preflight_payload["source_mode"] == "real"
    assert preflight_payload["splits"]["train"] > 0
    assert preflight_payload["splits"]["val"] > 0
    assert preflight_payload["splits"]["test"] > 0

    output_dir = tmp_path / f"{Path(config_name).stem}_real"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_case.py",
            "--config",
            config_name,
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["dataset"] == "RM_017_Ottawa19"
    assert payload["graph_path"] == graph_path
    assert payload["source_mode"] == "real"

    common_files = [
        "dag.json",
        "validated_dag.json",
        "compiled_dag_manifest.json",
        "resolved_splits.json",
        "resolved_dataset_manifest.json",
        "artifact_index.json",
        "workflow_state.json",
        "dag_quality_summary.json",
        "decision_side_outputs.json",
        "final_report.md",
    ]
    for filename in common_files + [expected_file]:
        assert (output_dir / filename).exists()

    if graph_path == "ml":
        for filename in (
            "feature_list.json",
            "feature_separability_summary.json",
            "dataset_level_runtime_trace.json",
            "metrics.json",
            "predictions.json",
            "importance.json",
            "similarity_artifacts.json",
        ):
            assert (output_dir / filename).exists()
        feature_pipeline = json.loads((output_dir / "feature_pipeline.json").read_text(encoding="utf-8"))
        assert "execution_nodes" in feature_pipeline
        assert "output_specs" in feature_pipeline
        assert feature_pipeline["output_policy"] == "terminal_only"
        separability_summary = json.loads((output_dir / "feature_separability_summary.json").read_text(encoding="utf-8"))
        assert separability_summary["dataset"] == "RM_017_Ottawa19"
        assert separability_summary["graph_path"] == "ml"
        assert "top_features" in separability_summary
        assert "aggregate_scores" in separability_summary
        assert "split_stability" in separability_summary
        runtime_trace = json.loads((output_dir / "dataset_level_runtime_trace.json").read_text(encoding="utf-8"))
        assert runtime_trace["graph_path"] == "ml"
        assert runtime_trace["splits"]
    else:
        for filename in ("training_curves.json", "checkpoint.json", "metrics.json", "importance.json", "similarity_artifacts.json"):
            assert (output_dir / filename).exists()
        model_build_plan = json.loads((output_dir / "model_build_plan.json").read_text(encoding="utf-8"))
        assert "execution_nodes" in model_build_plan
        assert "output_specs" in model_build_plan
        assert model_build_plan["output_policy"] == "terminal_only"

    resolved_manifest = json.loads((output_dir / "resolved_dataset_manifest.json").read_text(encoding="utf-8"))
    assert resolved_manifest["dataset_name"] == "RM_017_Ottawa19"
    assert resolved_manifest["source_mode"] == "real"

    resolved_splits = json.loads((output_dir / "resolved_splits.json").read_text(encoding="utf-8"))
    assert resolved_splits["train_ids"]
    assert resolved_splits["val_ids"]
    assert resolved_splits["test_ids"]

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert report_section in report_text


@pytest.mark.skipif(
    not all(path.exists() for path in (REAL_RM101_METADATA, REAL_RM101_H5)),
    reason="Real RM101 PHM-Vibench files are not available in this environment.",
)
@pytest.mark.parametrize(
    ("config_name", "graph_path", "expected_file", "report_section"),
    (
        ("config/runs/rm101_ml_test.yaml", "ml", "feature_pipeline.json", "## ML Evidence"),
        ("config/runs/rm101_torch_test.yaml", "torch", "model_build_plan.json", "## Torch Evidence"),
    ),
)
def test_real_rm101_ml_and_torch_smoke(tmp_path: Path, config_name: str, graph_path: str, expected_file: str, report_section: str):
    preflight = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", config_name],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    preflight_payload = json.loads(preflight.stdout)
    assert preflight_payload["dataset_name"] == "RM_101_THU_GEARBOX"
    assert preflight_payload["graph_path"] == graph_path
    assert preflight_payload["source_mode"] == "real"
    assert preflight_payload["splits"]["train"] > 0
    assert preflight_payload["splits"]["val"] > 0
    assert preflight_payload["splits"]["test"] > 0

    output_dir = tmp_path / f"{Path(config_name).stem}_real"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_case.py",
            "--config",
            config_name,
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["dataset"] == "RM_101_THU_GEARBOX"
    assert payload["graph_path"] == graph_path
    assert payload["source_mode"] == "real"

    common_files = [
        "dag.json",
        "validated_dag.json",
        "compiled_dag_manifest.json",
        "resolved_splits.json",
        "resolved_dataset_manifest.json",
        "artifact_index.json",
        "workflow_state.json",
        "dag_quality_summary.json",
        "decision_side_outputs.json",
        "final_report.md",
    ]
    for filename in common_files + [expected_file]:
        assert (output_dir / filename).exists()

    if graph_path == "ml":
        for filename in (
            "feature_list.json",
            "feature_separability_summary.json",
            "dataset_level_runtime_trace.json",
            "metrics.json",
            "predictions.json",
            "importance.json",
            "similarity_artifacts.json",
        ):
            assert (output_dir / filename).exists()
        feature_pipeline = json.loads((output_dir / "feature_pipeline.json").read_text(encoding="utf-8"))
        assert "execution_nodes" in feature_pipeline
        assert "output_specs" in feature_pipeline
        assert feature_pipeline["output_policy"] == "terminal_only"
        separability_summary = json.loads((output_dir / "feature_separability_summary.json").read_text(encoding="utf-8"))
        assert separability_summary["dataset"] == "RM_101_THU_GEARBOX"
        assert separability_summary["graph_path"] == "ml"
        assert "top_features" in separability_summary
        assert "aggregate_scores" in separability_summary
        assert "split_stability" in separability_summary
        runtime_trace = json.loads((output_dir / "dataset_level_runtime_trace.json").read_text(encoding="utf-8"))
        assert runtime_trace["graph_path"] == "ml"
        assert runtime_trace["splits"]
    else:
        for filename in ("training_curves.json", "checkpoint.json", "metrics.json", "importance.json", "similarity_artifacts.json"):
            assert (output_dir / filename).exists()
        model_build_plan = json.loads((output_dir / "model_build_plan.json").read_text(encoding="utf-8"))
        assert "execution_nodes" in model_build_plan
        assert "output_specs" in model_build_plan
        assert model_build_plan["output_policy"] == "terminal_only"

    resolved_manifest = json.loads((output_dir / "resolved_dataset_manifest.json").read_text(encoding="utf-8"))
    assert resolved_manifest["dataset_name"] == "RM_101_THU_GEARBOX"
    assert resolved_manifest["source_mode"] == "real"

    resolved_splits = json.loads((output_dir / "resolved_splits.json").read_text(encoding="utf-8"))
    assert resolved_splits["train_ids"]
    assert resolved_splits["val_ids"]
    assert resolved_splits["test_ids"]

    report_text = (output_dir / "final_report.md").read_text(encoding="utf-8")
    assert report_section in report_text
