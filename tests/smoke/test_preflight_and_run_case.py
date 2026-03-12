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
        assert (output_dir / "compiled_dag_manifest.json").exists()
        assert (output_dir / "resolved_splits.json").exists()
        assert (output_dir / "resolved_dataset_manifest.json").exists()
        assert (output_dir / "final_report.md").exists()
        assert (output_dir / expected_file).exists()


@pytest.mark.skipif(
    not all(path.exists() for path in (REAL_RM101_METADATA, REAL_RM101_H5, REAL_OTTAWA_METADATA, REAL_OTTAWA_H5)),
    reason="Real PHM-Vibench files are not available in this environment.",
)
def test_real_configs_preflight_and_dag_only(tmp_path: Path):
    for config_name, expected_dataset in (
        ("config/runs/rm101_dag.yaml", "RM_101_THU_GEARBOX"),
        ("config/runs/ottawa_dag.yaml", "RM_017_Ottawa19"),
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
        assert (output_dir / "resolved_splits.json").exists()
        assert (output_dir / "resolved_dataset_manifest.json").exists()
