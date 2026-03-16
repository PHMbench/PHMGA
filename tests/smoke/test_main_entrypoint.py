from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_main_entrypoint_supports_preflight_and_run_case(tmp_path: Path):
    preflight = subprocess.run(
        [sys.executable, "main.py", "runtime.action=preflight", "+runs=rm101_synth_dag"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    preflight_payload = json.loads(preflight.stdout)
    assert preflight_payload["status"] == "ok"
    assert preflight_payload["dataset_name"] == "RM101_SYNTH"
    assert preflight_payload["graph_path"] == "dag_only"

    output_dir = tmp_path / "main_rm101_synth_dag"
    run_case = subprocess.run(
        [
            sys.executable,
            "main.py",
            "+runs=rm101_synth_dag",
            f"runtime.output_dir={output_dir}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(run_case.stdout)
    assert payload["graph_path"] == "dag_only"
    assert (output_dir / "compiled_dag_manifest.json").exists()
    assert (output_dir / "final_report.md").exists()
