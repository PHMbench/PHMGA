from __future__ import annotations

from pathlib import Path

from scripts import run_case as run_case_module
from src.config import load_runtime_config
from src.dag import validate_dag_json


ROOT = Path(__file__).resolve().parents[2]


def test_run_case_uses_configured_experiment_user_instruction(tmp_path: Path, monkeypatch):
    runtime_config = load_runtime_config(
        ROOT / "config/runs/rm101_synth_dag.yaml",
        output_dir=str(tmp_path / "artifacts"),
    )
    runtime_config["experiment"]["user_instruction"] = "Config-owned instruction for smoke validation."
    captured: dict[str, str] = {}

    def fake_frontend_loop(state, protocol, llm, catalog, runtime_config):
        del protocol, llm, catalog, runtime_config
        captured["user_instruction"] = state.user_instruction
        state.dag = validate_dag_json(
            {
                "nodes": [
                    {
                        "node_id": "signal_input",
                        "op_uid": "input.signal",
                        "name": "Signal Input",
                        "kind": "input",
                        "operator_category": "INPUT",
                        "rank_class": "rank_same",
                        "params": {},
                        "parents": [],
                        "in_shape": [1, 1024],
                        "out_shape": [1, 1024],
                        "backend_availability": ["np", "pt", "sym"],
                        "execution_role": "fixed",
                        "legal_paths": ["dag_only", "ml", "torch"],
                        "input_bindings": {},
                        "plan_step_ref": "root",
                        "rationale": "Synthetic root for config contract test.",
                    }
                ],
                "edges": [],
            }
        )
        state.compiled_bundle = {"dag_hash": "config-contract-test", "path_type": state.graph_path}
        state.path_artifacts = {"method_description": "Config-contract test artifact."}
        state.final_report = "Config-contract test report."
        return state

    monkeypatch.setattr(run_case_module, "_run_frontend_loop", fake_frontend_loop)
    monkeypatch.setattr(run_case_module, "get_llm", lambda runtime_config: None)
    monkeypatch.setattr(run_case_module, "get_operator_catalog", lambda: None)

    payload = run_case_module.run_case(runtime_config)

    assert captured["user_instruction"] == "Config-owned instruction for smoke validation."
    assert payload["graph_path"] == "dag_only"
