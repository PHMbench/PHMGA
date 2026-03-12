from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_preflight_reports_two_datasets_and_three_paths():
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--config", "config/config.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert set(payload["datasets"]) == {"RM101", "Ottawa"}
    assert payload["graph_paths"] == ["dag_only", "ml", "torch"]


def test_run_case_all_dataset_path_pairs(tmp_path: Path):
    combos = [
        ("RM101", "dag_only", "dag_artifacts.json"),
        ("RM101", "ml", "feature_pipeline.json"),
        ("RM101", "torch", "model_build_plan.json"),
        ("Ottawa", "dag_only", "dag_artifacts.json"),
        ("Ottawa", "ml", "feature_pipeline.json"),
        ("Ottawa", "torch", "model_build_plan.json"),
    ]
    for dataset_name, graph_path, expected_file in combos:
        output_dir = tmp_path / f"{dataset_name.lower()}_{graph_path}"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/run_case.py",
                "--config",
                "config/config.yaml",
                "--dataset",
                dataset_name,
                "--graph-path",
                graph_path,
                "--output-dir",
                str(output_dir),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(proc.stdout)
        assert payload["dataset"] == dataset_name
        assert payload["graph_path"] == graph_path
        assert (output_dir / "dag.json").exists()
        assert (output_dir / "compiled_dag_manifest.json").exists()
        assert (output_dir / "final_report.md").exists()
        assert (output_dir / expected_file).exists()
