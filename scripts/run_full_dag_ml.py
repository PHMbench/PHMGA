from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOTENV_PATH = ROOT / ".env"

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

from src.config import load_case_config
from src.evaluation import evaluate_full_dag_leaves
from src.manual_workflow import export_dag_artifacts, generate_report_from_state
from src.states.phm_states import PHMState
from src.utils import load_state


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _write_json(path: str | Path, payload: Any) -> None:
    target = _ensure_parent(path)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = _ensure_parent(path)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_split_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = _ensure_parent(path)
    fieldnames = ["split", "label", "domain_id", "domain_description", "n_ids"]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _dict_to_markdown(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {title}"]
    if not rows:
        lines.append("_No rows available._")
        return "\n".join(lines)
    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            values.append("" if value is None else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _selection_markdown(
    dag_summary: dict[str, Any],
    final_selection: dict[str, Any],
    node_rows: list[dict[str, Any]],
    protocol_summary: dict[str, Any],
) -> str:
    best_single = dict(final_selection.get("best_single_leaf") or {})
    weighted = dict((final_selection.get("weighted_ensemble") or {}).get("metrics") or {})
    selection_basis = str(final_selection.get("selection_basis") or "")
    channel_aliases = dict(protocol_summary.get("channel_aliases") or {})
    n_train_windows = int(protocol_summary.get("n_train_windows", 0) or 0)
    n_val_windows = int(protocol_summary.get("n_val_windows", 0) or 0)
    n_test_windows = int(protocol_summary.get("n_test_windows", 0) or 0)
    lines = [
        "## Automated Selection",
        f"- Selection basis: `{selection_basis}`" if selection_basis else "- Selection basis: ``",
        f"- Best single leaf: `{best_single.get('leaf_id', '')}`",
        f"- Final choice: `{(final_selection.get('final_choice') or {}).get('strategy', '')}`",
        f"- Best single test accuracy: {best_single.get('test_accuracy', 0.0)}",
        f"- Best single test macro_f1: {best_single.get('test_macro_f1', 0.0)}",
        f"- Weighted ensemble test accuracy: {weighted.get('test_accuracy', weighted.get('accuracy', 0.0))}",
        f"- Weighted ensemble test macro_f1: {weighted.get('test_macro_f1', weighted.get('macro_f1', 0.0))}",
        "",
        "## DAG Summary",
        f"- Depth: {dag_summary.get('depth', 0)}",
        f"- Node count: {dag_summary.get('node_count', 0)}",
        f"- Edge count: {dag_summary.get('edge_count', 0)}",
        f"- Unique ops: {', '.join(dag_summary.get('unique_ops', [])) or 'none'}",
        "",
        "## Evaluation Protocol",
        f"- Dataset: `{protocol_summary.get('dataset_name', '')}`",
        f"- Window configuration: `{protocol_summary.get('window', {})}`",
        f"- Train / val / test windows: `{n_train_windows}` / `{n_val_windows}` / `{n_test_windows}`",
        "",
        _dict_to_markdown("Node Metrics", node_rows),
    ]
    if channel_aliases:
        alias_rows = [{"channel": channel, "alias": alias} for channel, alias in sorted(channel_aliases.items())]
        lines.extend(["", _dict_to_markdown("Channel Aliases", alias_rows)])
    return "\n".join(lines).strip() + "\n"


def _has_valid_leaf_metrics(node_rows: list[dict[str, Any]], final_selection: dict[str, Any]) -> bool:
    best_single = dict(final_selection.get("best_single_leaf") or {})
    if best_single.get("leaf_id") or best_single.get("node_id"):
        return True

    for row in node_rows:
        if row.get("failure_reason"):
            continue
        if row.get("leaf_id") or row.get("node_id"):
            return True
    return False


def _resolve_case_and_state(
    *,
    state_path: str | Path | None,
    case_name: str | None,
    config_root: str | Path | None,
    case_config: dict[str, Any] | None,
) -> tuple[PHMState, dict[str, Any], Path]:
    if state_path is None and case_config is None and case_name is None:
        raise ValueError("run_full_dag_ml requires state_path or case_name.")

    config = case_config or load_case_config(str(case_name), config_root=config_root)
    resolved_state_path = (
        Path(state_path).expanduser().resolve()
        if state_path is not None
        else Path(config["state_save_path"]).expanduser().resolve()
    )
    state = load_state(str(resolved_state_path))
    if state is None:
        raise RuntimeError(f"Failed to load state from {resolved_state_path}")
    return state, config, resolved_state_path


def run_full_dag_ml(
    *,
    state_path: str | Path | None = None,
    case_name: str | None = None,
    config_root: str | Path | None = None,
    case_config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    algorithm: str = "RandomForest",
    ensemble_method: str = "hard_voting",
    cv_folds: int = 5,
    candidate_algorithms: list[str] | None = None,
    parallel_workers: int | None = None,
    skip_llm_report: bool = False,
) -> dict[str, Any]:
    state, config, resolved_state_path = _resolve_case_and_state(
        state_path=state_path,
        case_name=case_name,
        config_root=config_root,
        case_config=case_config,
    )
    output_root = Path(output_dir).expanduser().resolve() if output_dir is not None else resolved_state_path.parent
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "run_manifest.json"
    failure_path = output_root / "failure.json"

    try:
        evaluation = evaluate_full_dag_leaves(
            state,
            case_config=config,
            algorithm=algorithm,
            ensemble_method=ensemble_method,
            cv_folds=cv_folds,
            candidate_algorithms=candidate_algorithms,
            parallel_workers=parallel_workers,
        )
    except Exception as exc:
        failure_payload = {
            "status": "failed",
            "case_name": case_name or state.case_name,
            "state_path": str(resolved_state_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(failure_path, failure_payload)
        raise

    replayed_state = evaluation["state"]
    ml_results = dict(evaluation["ml_results"] or {})
    dag_summary = dict(evaluation["dag_summary"] or {})
    node_rows = list(evaluation.get("leaf_metrics") or ml_results.get("node_level_results") or [])
    final_selection = dict(evaluation.get("final_selection") or {})
    selection_predictions = dict(evaluation.get("selection_predictions") or ml_results.get("selection_predictions") or {})
    split_manifest = dict(evaluation.get("split_manifest") or {})
    split_summary_by_label_domain = list(evaluation.get("split_summary_by_label_domain") or [])

    graphs_dir = output_root / "graphs"
    graph_artifacts = export_dag_artifacts(
        replayed_state,
        output_dir=graphs_dir,
        stem="dag",
        max_nodes=None,
        save_png=True,
        save_json=True,
    )

    metrics_md_path = output_root / "node_metrics.md"
    metrics_json_path = output_root / "node_metrics.json"
    metrics_csv_path = output_root / "node_metrics.csv"
    selection_path = output_root / "final_selection.json"
    dag_summary_path = output_root / "dag_summary.json"
    protocol_summary_path = output_root / "protocol_summary.json"
    split_manifest_path = output_root / "split_manifest.json"
    split_summary_by_label_domain_path = output_root / "split_summary_by_label_domain.csv"
    ml_results_path = output_root / "ml_results.pkl"
    selection_predictions_path = output_root / "selection_predictions.pkl"
    report_path = output_root / "final_report.md"

    _write_json(metrics_json_path, node_rows)
    _write_csv(metrics_csv_path, node_rows)
    _ensure_parent(metrics_md_path).write_text(_dict_to_markdown("Node Metrics", node_rows), encoding="utf-8")
    _write_json(selection_path, final_selection)
    _write_json(dag_summary_path, dag_summary)
    _write_json(protocol_summary_path, dict(ml_results.get("protocol_summary") or {}))
    _write_json(split_manifest_path, split_manifest)
    _write_split_summary_csv(split_summary_by_label_domain_path, split_summary_by_label_domain)
    with _ensure_parent(ml_results_path).open("wb") as handle:
        pickle.dump(ml_results, handle)
    with _ensure_parent(selection_predictions_path).open("wb") as handle:
        pickle.dump(selection_predictions, handle)

    protocol_summary = dict(ml_results.get("protocol_summary") or {})
    summary_md = _selection_markdown(dag_summary, final_selection, node_rows, protocol_summary)
    ml_results["metrics_markdown"] = str(ml_results.get("metrics_markdown") or "")
    ml_results["metrics_markdown"] = (ml_results["metrics_markdown"].rstrip() + "\n\n" + summary_md).strip()
    report_error: dict[str, Any] | None = None
    evaluation_failure: dict[str, Any] | None = None
    has_valid_leaf_metrics = _has_valid_leaf_metrics(node_rows, final_selection)
    if not has_valid_leaf_metrics:
        evaluation_failure = {
            "status": "failed",
            "stage": "full_dag_ml",
            "error_type": "NoValidTerminalLeaves",
            "error": "No processed terminal leaves produced valid shallow ML metrics.",
            "leaf_count": len(node_rows),
            "failed_leaf_count": sum(1 for row in node_rows if row.get("failure_reason")),
            "sample_failure_reasons": sorted(
                {
                    str(row.get("failure_reason"))
                    for row in node_rows
                    if row.get("failure_reason")
                }
            ),
        }
        report_markdown = (
            "## Report Generation Skipped\n"
            "- note: no valid terminal leaf produced shallow ML metrics, so LLM report generation was skipped.\n"
        )
    elif skip_llm_report:
        report_markdown = (
            "## Report Generation Skipped\n"
            "- note: skip_llm_report=true, so this file contains the deterministic experiment summary only.\n"
        )
    else:
        try:
            report_result = generate_report_from_state(replayed_state, ml_results=ml_results, report_path=None)
            report_markdown = str(report_result.get("final_report") or "").strip()
        except Exception as exc:
            report_markdown = (
                "## Report Generation Fallback\n"
                f"- error_type: `{type(exc).__name__}`\n"
                f"- error: {str(exc)}\n"
                "- note: LLM report generation failed, so this file falls back to the deterministic experiment summary.\n"
            )
            report_error = {
                "status": "failed",
                "stage": "report",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

    final_report = (summary_md + "\n" + report_markdown).strip() + "\n"
    _ensure_parent(report_path).write_text(final_report, encoding="utf-8")
    if report_error is not None:
        _write_json(failure_path, report_error)
    elif evaluation_failure is not None:
        _write_json(failure_path, evaluation_failure)

    llm_cfg = dict((getattr(replayed_state, "runtime_config", {}) or {}).get("llm") or {})
    run_metadata = dict((getattr(replayed_state, "runtime_config", {}) or {}).get("run_metadata") or {})
    status = "failed" if evaluation_failure is not None else "ok"
    manifest = {
        "status": status,
        "case_name": state.case_name or case_name,
        "provider": str(llm_cfg.get("provider", "")),
        "model": str(llm_cfg.get("model", "")),
        "run_type": str(run_metadata.get("run_type", "")),
        "paper_label": str(run_metadata.get("paper_label", "")),
        "state_path": str(resolved_state_path),
        "output_dir": str(output_root),
        "graph_artifacts": graph_artifacts,
        "dag_summary_path": str(dag_summary_path),
        "node_metrics_json_path": str(metrics_json_path),
        "node_metrics_csv_path": str(metrics_csv_path),
        "node_metrics_md_path": str(metrics_md_path),
        "selection_path": str(selection_path),
        "protocol_summary_path": str(protocol_summary_path),
        "split_manifest_path": str(split_manifest_path),
        "split_summary_by_label_domain_path": str(split_summary_by_label_domain_path),
        "ml_results_path": str(ml_results_path),
        "selection_predictions_path": str(selection_predictions_path),
        "report_path": str(report_path),
        "dag_summary": dag_summary,
        "final_selection": final_selection,
        "protocol_summary": protocol_summary,
        "channel_aliases": dict(protocol_summary.get("channel_aliases") or {}),
        "channel_alias_summary": str(protocol_summary.get("channel_alias_summary") or ""),
        "algorithm": algorithm,
        "candidate_algorithms": list(ml_results.get("candidate_algorithms") or candidate_algorithms or []),
        "ensemble_method": ensemble_method,
        "cv_folds": cv_folds,
        "parallel_workers": int(ml_results.get("parallel_workers") or parallel_workers or 0),
        "selection_predictions": {
            "path": str(selection_predictions_path),
            "selection_basis": str(selection_predictions.get("selection_basis") or ""),
            "final_choice_strategy": str(selection_predictions.get("final_choice_strategy") or ""),
        },
        "report_error": report_error,
        "evaluation_failure": evaluation_failure,
        "skip_llm_report": bool(skip_llm_report),
    }
    _write_json(manifest_path, manifest)

    return {
        "status": status,
        "case_name": manifest["case_name"],
        "state_path": manifest["state_path"],
        "output_dir": manifest["output_dir"],
        "graph_artifacts": graph_artifacts,
        "dag_summary": dag_summary,
        "selection": final_selection,
        "node_metrics_path": str(metrics_json_path),
        "selection_predictions_path": str(selection_predictions_path),
        "report_path": str(report_path),
        "manifest_path": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run full-DAG terminal leaf shallow ML and reporting.")
    parser.add_argument("--state-path", default=None, help="Path to a saved builder state pickle.")
    parser.add_argument("--case", default=None, help="Case name under config/ if --state-path is omitted.")
    parser.add_argument("--config-root", default=None, help="Alternative config root when using --case.")
    parser.add_argument("--output-dir", default=None, help="Directory for graph, ML, and report artifacts.")
    parser.add_argument("--algorithm", default="RandomForest", help="Shallow ML algorithm.")
    parser.add_argument("--ensemble-method", default="hard_voting", help="Weighted ensemble strategy.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--candidate-algorithms", default="", help="Comma-separated shallow models to evaluate per leaf.")
    parser.add_argument("--parallel-workers", type=int, default=0, help="Parallel workers for leaf x model evaluation. 0 keeps serial execution.")
    parser.add_argument("--skip-llm-report", action="store_true", help="Skip LLM report generation and write deterministic summaries only.")
    args = parser.parse_args(argv)

    if load_dotenv is not None and DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH)

    candidate_algorithms = [item.strip() for item in str(args.candidate_algorithms).split(",") if item.strip()]
    try:
        payload = run_full_dag_ml(
            state_path=args.state_path,
            case_name=args.case,
            config_root=args.config_root,
            output_dir=args.output_dir,
            algorithm=args.algorithm,
            ensemble_method=args.ensemble_method,
            cv_folds=args.cv_folds,
            candidate_algorithms=candidate_algorithms or None,
            parallel_workers=args.parallel_workers or None,
            skip_llm_report=args.skip_llm_report,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
