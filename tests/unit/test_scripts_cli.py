from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.export_paper_phmga import export_paper_bundle
from scripts.run_case import run_case_cli
from scripts.run_full_dag_ml import run_full_dag_ml
from scripts.run_simulated_rm101 import run_simulated_rm101
from src.config import load_case_config
from src.states.phm_states import DAGState, InputData, PHMState, ProcessedData


def _build_state(tmp_path: Path) -> PHMState:
    ref_features = tmp_path / "ref_features.npz"
    tst_features = tmp_path / "tst_features.npz"
    np.savez(ref_features, ref_a=np.array([1.0, 0.0]), ref_b=np.array([0.0, 1.0]))
    np.savez(tst_features, test_a=np.array([0.9, 0.1]), test_b=np.array([0.2, 0.8]))

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
        runtime_config={"llm": {"provider": "openrouter", "model": "z-ai/glm-4.5-air:free"}},
    )


def test_run_case_cli_writes_resolved_config_and_forwards_overrides(make_case_config):
    case = make_case_config(case_name="case_script_run_case", graph="builder")
    run_dir = Path(case["state_save_path"]).parent / "script_run"

    seen = {}

    def fake_run_case(case_name: str, *, config_root=None):
        seen["case_name"] = case_name
        seen["config_root"] = Path(config_root)
        cfg = load_case_config(case_name, config_root=config_root)
        assert cfg["builder"]["graph"] == "executor"
        assert cfg["llm"]["provider"] == "bigmodel"
        assert cfg["llm"]["model"] == "glm-4.7-flashx"
        assert Path(cfg["save_dir"]).resolve() == run_dir.resolve()
        assert Path(cfg["state_save_path"]).name == f"{case['case_name']}_built_state.pkl"
        assert Path(cfg["report_path"]).name == f"{case['case_name']}_final_report.md"
        return {
            "status": "ok",
            "graph": cfg["builder"]["graph"],
            "state_save_path": cfg["state_save_path"],
            "report_path": cfg["report_path"],
            "has_final_report": False,
        }

    import scripts.run_case as run_case_module

    original = run_case_module.run_graph_case
    run_case_module.run_graph_case = fake_run_case
    try:
        payload = run_case_cli(
            case["case_name"],
            config_root=case["config_root"],
            graph="executor",
            provider="bigmodel",
            model="glm-4.7-flashx",
            run_dir=run_dir,
        )
    finally:
        run_case_module.run_graph_case = original

    assert seen["case_name"] == case["case_name"]
    assert seen["config_root"] == run_dir / "config"
    assert payload["graph"] == "executor"
    assert Path(payload["resolved_config_path"]).exists()
    assert Path(payload["run_manifest_path"]).exists()
    resolved = yaml.safe_load(Path(payload["resolved_config_path"]).read_text(encoding="utf-8"))
    assert resolved["builder"]["graph"] == "executor"
    assert resolved["llm"]["provider"] == "bigmodel"


def test_run_full_dag_ml_writes_node_metrics_and_final_report(tmp_path: Path, monkeypatch):
    state = _build_state(tmp_path)
    state_path = tmp_path / "builder_state.pkl"
    with state_path.open("wb") as fh:
        pickle.dump(state, fh)

    case_config = {
        "name": "demo_case",
        "state_save_path": str(state_path),
        "data": {
            "dataset_name": "RM101",
            "metadata_path": str(tmp_path / "missing.xlsx"),
            "h5_path": str(tmp_path / "missing.h5"),
            "selection": {"dataset_id": 101},
            "split": {"strategy": "stratified_fixed_per_class", "train_per_class": 1, "val_per_class": 1, "test_per_class": 1},
            "window": {"window_size": 8, "stride": 8, "slice_mode": "centered", "drop_last_window": False},
        },
    }

    def fake_evaluate_full_dag_leaves(
        state_obj,
        *,
        case_config=None,
        split_records=None,
        cv_folds=5,
        algorithm="RandomForest",
        ensemble_method="hard_voting",
        candidate_algorithms=None,
        parallel_workers=None,
    ):
        assert case_config is not None
        return {
            "state": state_obj,
            "dag_summary": {"depth": 2, "node_count": 2, "edge_count": 1, "unique_ops": ["fft"]},
            "leaf_metrics": [
                {
                    "node_id": "fft_01_ch1",
                    "leaf_id": "fft_01_ch1",
                    "branch_id": "fft_01_ch1",
                    "branch_summary": "fft",
                    "feature_dim": 2,
                    "train_accuracy": 1.0,
                    "train_macro_f1": 1.0,
                    "val_accuracy": 1.0,
                    "val_macro_f1": 1.0,
                    "test_accuracy": 1.0,
                    "test_macro_f1": 0.9,
                    "selected_single_best": True,
                    "selected_in_ensemble": True,
                    "failure_reason": "",
                }
            ],
            "final_selection": {
                "best_single_leaf": {
                    "leaf_id": "fft_01_ch1",
                    "test_accuracy": 1.0,
                    "test_macro_f1": 0.9,
                },
                "weighted_ensemble": {"metrics": {"accuracy": 1.0, "macro_f1": 0.9, "test_accuracy": 1.0, "test_macro_f1": 0.9}},
                "final_choice": {"strategy": "best_single_leaf"},
                "selection_basis": "val_macro_f1",
            },
            "selection_predictions": {
                "selection_basis": "val_macro_f1",
                "window_ids_val": np.asarray(["val_a", "val_b"], dtype=object),
                "window_ids_test": np.asarray(["test_a", "test_b"], dtype=object),
                "y_val": np.asarray([0, 1]),
                "y_test": np.asarray([0, 1]),
                "best_single_leaf": {
                    "val_pred": np.asarray([0, 1]),
                    "test_pred": np.asarray([0, 1]),
                    "val_macro_f1": 1.0,
                },
                "weighted_ensemble": {
                    "val_pred": np.asarray([0, 1]),
                    "test_pred": np.asarray([0, 1]),
                    "val_macro_f1": 1.0,
                },
                "final_choice": {
                    "strategy": "best_single_leaf",
                    "val_pred": np.asarray([0, 1]),
                    "test_pred": np.asarray([0, 1]),
                    "val_macro_f1": 1.0,
                },
                "final_choice_strategy": "best_single_leaf",
            },
            "ml_results": {
                "models": {"fft_01_ch1": {"metrics": {"test_accuracy": 1.0, "test_macro_f1": 0.9}}},
                "node_level_results": [
                    {
                        "node_id": "fft_01_ch1",
                        "leaf_id": "fft_01_ch1",
                        "feature_dim": 2,
                        "test_accuracy": 1.0,
                        "test_macro_f1": 0.9,
                    }
                ],
                "final_selection": {
                    "best_single_leaf": "fft_01_ch1",
                    "best_single_leaf_metrics": {"test_accuracy": 1.0, "test_macro_f1": 0.9},
                    "final_choice": "best_single_leaf",
                    "selection_basis": "val_macro_f1",
                },
                "weighted_ensemble_metrics": {"accuracy": 1.0, "macro_f1": 0.9, "test_accuracy": 1.0, "test_macro_f1": 0.9},
                "selection_predictions": {
                    "selection_basis": "val_macro_f1",
                    "window_ids_val": np.asarray(["val_a", "val_b"], dtype=object),
                    "window_ids_test": np.asarray(["test_a", "test_b"], dtype=object),
                    "y_val": np.asarray([0, 1]),
                    "y_test": np.asarray([0, 1]),
                    "best_single_leaf": {
                        "val_pred": np.asarray([0, 1]),
                        "test_pred": np.asarray([0, 1]),
                        "val_macro_f1": 1.0,
                    },
                    "weighted_ensemble": {
                        "val_pred": np.asarray([0, 1]),
                        "test_pred": np.asarray([0, 1]),
                        "val_macro_f1": 1.0,
                    },
                    "final_choice": {
                        "strategy": "best_single_leaf",
                        "val_pred": np.asarray([0, 1]),
                        "test_pred": np.asarray([0, 1]),
                        "val_macro_f1": 1.0,
                    },
                    "final_choice_strategy": "best_single_leaf",
                },
                "metrics_markdown": "| node_id | test_accuracy |\n|---|---|\n| fft_01_ch1 | 1.0 |\n",
                "dag_summary": {"depth": 2, "node_count": 2, "edge_count": 1, "unique_ops": ["fft"]},
                "protocol_summary": {"dataset_name": "RM101", "n_train_windows": 4, "n_val_windows": 2, "n_test_windows": 2},
            },
        }

    def fake_generate_report_from_state(state_obj, *, ml_results=None, report_path=None):
        assert "final_selection" in ml_results
        report = "# Report\nBody"
        return {"final_report": report}

    monkeypatch.setattr("scripts.run_full_dag_ml.evaluate_full_dag_leaves", fake_evaluate_full_dag_leaves)
    monkeypatch.setattr("scripts.run_full_dag_ml.generate_report_from_state", fake_generate_report_from_state)

    output_dir = tmp_path / "run_out"
    payload = run_full_dag_ml(state_path=state_path, output_dir=output_dir, cv_folds=2, case_config=case_config)

    assert payload["status"] == "ok"
    assert Path(payload["report_path"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert (output_dir / "graphs" / "dag.json").exists()
    assert (output_dir / "graphs" / "dag.png").exists() or (output_dir / "graphs" / "dag.dot").exists()
    assert (output_dir / "node_metrics.json").exists()
    assert (output_dir / "node_metrics.csv").exists()
    assert (output_dir / "node_metrics.md").exists()
    assert (output_dir / "final_selection.json").exists()
    assert (output_dir / "dag_summary.json").exists()
    assert (output_dir / "selection_predictions.pkl").exists()

    node_metrics = json.loads((output_dir / "node_metrics.json").read_text(encoding="utf-8"))
    assert node_metrics[0]["node_id"] == "fft_01_ch1"
    assert node_metrics[0]["test_accuracy"] == 1.0
    assert node_metrics[0]["test_macro_f1"] == 0.9

    selection = json.loads((output_dir / "final_selection.json").read_text(encoding="utf-8"))
    assert selection["best_single_leaf"]["leaf_id"] == "fft_01_ch1"
    assert selection["final_choice"]["strategy"] in {"best_single_leaf", "weighted_ensemble"}
    assert selection["selection_basis"] == "val_macro_f1"
    protocol_summary = json.loads((output_dir / "protocol_summary.json").read_text(encoding="utf-8"))
    assert protocol_summary["n_train_windows"] == 4
    assert "Automated Selection" in Path(payload["report_path"]).read_text(encoding="utf-8")


def test_run_full_dag_ml_marks_failed_when_no_valid_terminal_leaf(tmp_path: Path, monkeypatch):
    state = _build_state(tmp_path)
    state_path = tmp_path / "builder_state.pkl"
    with state_path.open("wb") as fh:
        pickle.dump(state, fh)

    case_config = {
        "name": "demo_case",
        "state_save_path": str(state_path),
        "data": {
            "dataset_name": "RM101",
            "metadata_path": str(tmp_path / "missing.xlsx"),
            "h5_path": str(tmp_path / "missing.h5"),
            "selection": {"dataset_id": 101},
            "split": {"strategy": "stratified_fixed_per_class", "train_per_class": 1, "val_per_class": 1, "test_per_class": 1},
            "window": {"window_size": 8, "stride": 8, "slice_mode": "centered", "drop_last_window": False},
        },
    }

    def fake_evaluate_full_dag_leaves(
        state_obj,
        *,
        case_config=None,
        split_records=None,
        cv_folds=5,
        algorithm="RandomForest",
        ensemble_method="hard_voting",
        candidate_algorithms=None,
        parallel_workers=None,
    ):
        return {
            "state": state_obj,
            "dag_summary": {"depth": 1, "node_count": 1, "edge_count": 0, "unique_ops": []},
            "leaf_metrics": [
                {
                    "leaf_id": "ch1",
                    "branch_id": "ch1",
                    "branch_summary": "ch1",
                    "failure_reason": "input leaf is not a processed feature branch",
                }
            ],
            "final_selection": {
                "best_single_leaf": {},
                "weighted_ensemble": {"metrics": {"accuracy": 0.0, "macro_f1": 0.0}},
                "final_choice": {"strategy": ""},
            },
            "ml_results": {
                "node_level_results": [
                    {
                        "leaf_id": "ch1",
                        "branch_id": "ch1",
                        "branch_summary": "ch1",
                        "failure_reason": "input leaf is not a processed feature branch",
                    }
                ],
                "metrics_markdown": "",
                "dag_summary": {"depth": 1, "node_count": 1, "edge_count": 0, "unique_ops": []},
                "protocol_summary": {"dataset_name": "RM101"},
            },
        }

    def fake_generate_report_from_state(state_obj, *, ml_results=None, report_path=None):
        return {"final_report": "# Report\nNo valid leaf"}

    monkeypatch.setattr("scripts.run_full_dag_ml.evaluate_full_dag_leaves", fake_evaluate_full_dag_leaves)
    monkeypatch.setattr("scripts.run_full_dag_ml.generate_report_from_state", fake_generate_report_from_state)

    output_dir = tmp_path / "run_out_failed"
    payload = run_full_dag_ml(state_path=state_path, output_dir=output_dir, cv_folds=2, case_config=case_config)

    assert payload["status"] == "failed"
    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "full_dag_ml"
    assert failure["error_type"] == "NoValidTerminalLeaves"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"


def test_export_paper_bundle_aggregates_multiple_run_dirs(tmp_path: Path):
    root = tmp_path / "artifacts" / "rm101"
    run_a = root / "openrouter__glm45"
    run_b = root / "bigmodel__glm47"
    for run_dir, provider, model in (
        (run_a, "openrouter", "z-ai/glm-4.5-air:free"),
        (run_b, "bigmodel", "glm-4.7-flashx"),
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "graphs").mkdir(parents=True, exist_ok=True)
        (run_dir / "graphs" / "dag.png").write_bytes(b"fakepng")
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "case_name": "case_exp2",
                    "provider": provider,
                    "model": model,
                    "status": "ok",
                    "state_path": str(run_dir / "builder_state.pkl"),
                    "report_path": str(run_dir / "final_report.md"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "dag_summary.json").write_text(
            json.dumps({"depth": 5, "node_count": 7, "edge_count": 6, "unique_ops": ["fft", "mean"]}, indent=2),
            encoding="utf-8",
        )
        (run_dir / "node_metrics.json").write_text(
            json.dumps(
                [
                    {
                        "node_id": "fft_01",
                        "test_accuracy": 1.0,
                        "test_macro_f1": 0.9,
                        "cv_accuracy": 0.8,
                        "cv_macro_f1": 0.85,
                        "feature_dim": 2,
                        "train_samples": 2,
                        "test_samples": 2,
                        "branch_summary": "train=2, test=2",
                        "selected_single_best": True,
                        "selected_in_ensemble": True,
                        "failure_reason": "",
                    }
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "final_selection.json").write_text(
            json.dumps(
                {
                    "best_single_leaf": {
                        "leaf_id": "fft_01",
                        "test_accuracy": 1.0,
                        "test_macro_f1": 0.9,
                    },
                    "weighted_ensemble": {"metrics": {"accuracy": 1.0, "macro_f1": 0.91}},
                    "final_choice": {"strategy": "best_single_leaf"},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "final_report.md").write_text("# Report\n", encoding="utf-8")

    output_dir = tmp_path / "paper"
    payload = export_paper_bundle(root=root, output_dir=output_dir)

    assert Path(payload["backend_comparison_csv"]).exists()
    assert Path(payload["backend_comparison_md"]).exists()
    assert Path(payload["dag_summary_csv"]).exists()
    assert Path(payload["node_level_results_csv"]).exists()
    assert Path(payload["rm101_final_accuracy_csv"]).exists()
    assert Path(payload["analysis_path"]).exists()
    assert any(path.suffix in {".png", ".svg", ".dot"} for path in (output_dir / "figures").iterdir())

    with Path(payload["backend_comparison_csv"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["provider"] for row in rows} == {"openrouter", "bigmodel"}


def test_run_simulated_rm101_stages_runs_and_writes_bundle(tmp_path: Path, monkeypatch):
    simulated_root = tmp_path / "simulated"
    compare_root = tmp_path / "compare"
    paper_root = tmp_path / "paper"

    def fake_build_and_run_simulated(
        model_tag: str,
        *,
        case_name: str,
        output_root: Path,
        parallel_workers: int,
        run_suffix: str = "simulated_paper_v3",
    ):
        run_dir = output_root / model_tag.replace("/", "__")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "graphs").mkdir(exist_ok=True)
        (run_dir / "graphs" / "dag.png").write_bytes(b"png")
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "provider": "simulated",
                    "model": model_tag,
                    "paper_label": f"{model_tag} (simulated)",
                    "run_type": "simulated_variant",
                    "selection_predictions_path": str(run_dir / "selection_predictions.pkl"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (run_dir / "selection_predictions.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "selection_basis": "val_macro_f1",
                    "window_ids_val": np.asarray(["v1"], dtype=object),
                    "window_ids_test": np.asarray(["t1"], dtype=object),
                    "y_val": np.asarray([0]),
                    "y_test": np.asarray([0]),
                    "final_choice": {"strategy": "best_single_leaf", "val_pred": np.asarray([0]), "test_pred": np.asarray([0]), "val_macro_f1": 1.0},
                    "final_choice_strategy": "best_single_leaf",
                },
                handle,
            )
        (run_dir / "dag_summary.json").write_text(json.dumps({"depth": 3, "node_count": 20, "edge_count": 12, "unique_ops": ["fft", "rms"]}), encoding="utf-8")
        (run_dir / "node_metrics.json").write_text(json.dumps([]), encoding="utf-8")
        (run_dir / "final_selection.json").write_text(json.dumps({"selection_basis": "val_macro_f1"}), encoding="utf-8")
        (run_dir / "final_report.md").write_text("# Report\n", encoding="utf-8")
        (run_dir / "builder_state.pkl").write_bytes(b"state")
        return run_dir

    def fake_stage_simulated_runs(run_dirs, compare_root: Path):
        staged = []
        for run_dir in run_dirs:
            target = compare_root / Path(run_dir).name
            shutil.copytree(run_dir, target)
            staged.append(target)
        return staged

    def fake_stage_bigmodel_baseline(compare_root: Path, case_config: dict[str, Any], *, parallel_workers: int):
        run_dir = compare_root / "bigmodel__glm-4.7-flashx__real_baseline"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "provider": "bigmodel",
                    "model": "glm-4.7-flashx",
                    "paper_label": "BigModel / GLM-4.7-FlashX (real baseline)",
                    "run_type": "real_baseline",
                    "selection_predictions_path": str(run_dir / "selection_predictions.pkl"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        with (run_dir / "selection_predictions.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "selection_basis": "val_macro_f1",
                    "window_ids_val": np.asarray(["v1"], dtype=object),
                    "window_ids_test": np.asarray(["t1"], dtype=object),
                    "y_val": np.asarray([0]),
                    "y_test": np.asarray([0]),
                    "final_choice": {"strategy": "best_single_leaf", "val_pred": np.asarray([0]), "test_pred": np.asarray([0]), "val_macro_f1": 1.0},
                    "final_choice_strategy": "best_single_leaf",
                },
                handle,
            )
        return run_dir

    monkeypatch.setattr("scripts.run_simulated_rm101._build_and_run_simulated", fake_build_and_run_simulated)
    monkeypatch.setattr("scripts.run_simulated_rm101._stage_simulated_runs", fake_stage_simulated_runs)
    monkeypatch.setattr("scripts.run_simulated_rm101._stage_bigmodel_baseline", fake_stage_bigmodel_baseline)
    monkeypatch.setattr(
        "scripts.run_simulated_rm101.validate_complexity_ladder",
        lambda states: {"google/gemini-2.0-flash-001": {"depth": 3, "node_count": 20}},
    )
    monkeypatch.setattr(
        "scripts.run_simulated_rm101.export_paper_bundle",
        lambda *, root, output_dir: {"root": str(root), "output_dir": str(output_dir)},
    )
    monkeypatch.setattr(
        "scripts.run_simulated_rm101.compute_cross_dag_late_fusion",
        lambda run_dirs: {"selection_basis": "val_macro_f1", "fusion_method": "weighted_vote", "weights": [], "val_metrics": {"accuracy": 1.0, "macro_f1": 1.0}, "test_metrics": {"accuracy": 1.0, "macro_f1": 1.0}, "n_val_windows": 1, "n_test_windows": 1},
    )

    payload = run_simulated_rm101(
        case_name="case_exp2_paper",
        model_tag="google/gemini-2.0-flash-001",
        output_root=simulated_root,
        compare_root=compare_root,
        paper_output_dir=paper_root,
        include_bigmodel_baseline=True,
    )

    assert payload["include_bigmodel_baseline"] is True
    assert (paper_root / "cross_dag_late_fusion.json").exists()
    assert (paper_root / "simulated_rm101_bundle.json").exists()
