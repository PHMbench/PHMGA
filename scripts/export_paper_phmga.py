from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOTENV_PATH = ROOT / ".env"

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _load_node_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "node_metrics.json",
        run_dir / "node_metrics.csv",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            payload = _read_json(candidate)
            return payload if isinstance(payload, list) else []
        if pd is not None:
            try:
                frame = pd.read_csv(candidate)
                return frame.to_dict(orient="records")
            except Exception:
                continue
        with candidate.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader)
    return []


def _load_run_record(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        return None

    dag_summary = _load_optional_json(run_dir / "dag_summary.json") or {}
    selection = _load_optional_json(run_dir / "final_selection.json") or {}
    node_rows = _load_node_rows(run_dir)
    report_path = Path(manifest.get("report_path") or run_dir / "final_report.md")
    report_exists = report_path.exists()
    report_size = report_path.stat().st_size if report_exists else 0

    if not selection and node_rows:
        best_row = max(
            node_rows,
            key=lambda row: (
                float(row.get("cv_macro_f1", 0) or 0),
                float(row.get("cv_accuracy", 0) or 0),
                float(row.get("test_macro_f1", 0) or 0),
                -int(float(row.get("feature_dim", 0) or 0)),
                str(row.get("node_id", "")),
            ),
        )
        selection = {
            "best_single_leaf": best_row,
            "weighted_ensemble": {"metrics": {}},
            "final_choice": "best_single_leaf",
        }

    return {
        "run_dir": str(run_dir),
        "case_name": manifest.get("case_name", ""),
        "provider": manifest.get("provider", ""),
        "model": manifest.get("model", ""),
        "run_type": manifest.get("run_type", ""),
        "paper_label": manifest.get("paper_label", ""),
        "status": manifest.get("status", "unknown"),
        "state_path": manifest.get("state_path", ""),
        "report_path": str(report_path),
        "report_exists": report_exists,
        "report_size": report_size,
        "dag_depth": dag_summary.get("depth", 0),
        "node_count": dag_summary.get("node_count", 0),
        "edge_count": dag_summary.get("edge_count", 0),
        "unique_ops": ", ".join(dag_summary.get("unique_ops", [])) if dag_summary else "",
        "best_single_leaf": (selection.get("best_single_leaf", {}) or {}).get("leaf_id", (selection.get("best_single_leaf", {}) or {}).get("node_id", "")),
        "best_single_test_accuracy": (selection.get("best_single_leaf", {}) or {}).get("test_accuracy", ""),
        "best_single_test_macro_f1": (selection.get("best_single_leaf", {}) or {}).get("test_macro_f1", ""),
        "ensemble_test_accuracy": (selection.get("weighted_ensemble", {}) or {}).get("metrics", {}).get("test_accuracy", (selection.get("weighted_ensemble", {}) or {}).get("metrics", {}).get("accuracy", "")),
        "ensemble_test_macro_f1": (selection.get("weighted_ensemble", {}) or {}).get("metrics", {}).get("test_macro_f1", (selection.get("weighted_ensemble", {}) or {}).get("metrics", {}).get("macro_f1", "")),
        "final_choice": (selection.get("final_choice", {}) or {}).get("strategy", selection.get("final_choice", "")),
        "channel_aliases": manifest.get("channel_aliases", {}) or {},
        "channel_alias_summary": manifest.get("channel_alias_summary", ""),
        "node_rows": node_rows,
    }


def _markdown_table(rows: list[dict[str, Any]], headers: list[str] | None = None) -> str:
    if not rows:
        return "_No rows available._"
    headers = headers or list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            values.append("" if value is None else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_table_artifacts(rows: list[dict[str, Any]], base_path: Path) -> tuple[Path, Path]:
    csv_path = _ensure_parent(base_path.with_suffix(".csv"))
    md_path = _ensure_parent(base_path.with_suffix(".md"))
    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        md_path.write_text(pd.DataFrame(rows).to_markdown(index=False), encoding="utf-8")
    else:
        headers = list(rows[0].keys()) if rows else []
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        md_path.write_text(_markdown_table(rows, headers=headers), encoding="utf-8")
    return csv_path, md_path


def export_paper_bundle(
    *,
    root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir = output_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    for manifest_path in root_path.rglob("run_manifest.json"):
        run_dirs.append(manifest_path.parent)
    run_dirs = sorted({path.resolve() for path in run_dirs})

    records = []
    node_rows = []
    for run_dir in run_dirs:
        record = _load_run_record(run_dir)
        if record is None:
            continue
        records.append({k: v for k, v in record.items() if k != "node_rows"})
        for row in record.get("node_rows", []):
            node_rows.append(
                {
                    "case_name": record.get("case_name", ""),
                    "provider": record.get("provider", ""),
                    "model": record.get("model", ""),
                    **row,
                }
            )
        graph_candidates = [
            run_dir / "graphs" / "dag.png",
            run_dir / "graphs" / "dag.svg",
            run_dir / "graphs" / "dag.dot",
            run_dir / "graphs" / "final_dag.png",
            run_dir / "graphs" / "final_dag.svg",
            run_dir / "graphs" / "final_dag.dot",
        ]
        for candidate in graph_candidates:
            if candidate.exists() and candidate.stat().st_size > 0:
                target = figures_dir / f"{run_dir.name}{candidate.suffix}"
                shutil.copy2(candidate, target)
                break

    backend_csv, backend_md = _write_table_artifacts(records, output_path / "backend_comparison")
    dag_summary_csv, dag_summary_md = _write_table_artifacts(
        [
            {
                "case_name": record.get("case_name", ""),
                "paper_label": record.get("paper_label", ""),
                "run_type": record.get("run_type", ""),
                "provider": record.get("provider", ""),
                "model": record.get("model", ""),
                "dag_depth": record.get("dag_depth", 0),
                "node_count": record.get("node_count", 0),
                "edge_count": record.get("edge_count", 0),
                "unique_ops": record.get("unique_ops", ""),
                "final_choice": record.get("final_choice", ""),
            }
            for record in records
        ],
        output_path / "dag_summary_table",
    )
    node_csv, node_md = _write_table_artifacts(node_rows, output_path / "node_level_results")
    alias_rows = []
    for record in records:
        aliases = dict(record.get("channel_aliases") or {})
        if not aliases:
            continue
        for channel, alias in sorted(aliases.items()):
            alias_rows.append(
                {
                    "case_name": record.get("case_name", ""),
                    "paper_label": record.get("paper_label", ""),
                    "run_type": record.get("run_type", ""),
                    "channel": channel,
                    "alias": alias,
                }
            )
    alias_csv, alias_md = _write_table_artifacts(alias_rows, output_path / "channel_alias_table")
    final_accuracy_csv, final_accuracy_md = _write_table_artifacts(
        [
            {
                "case_name": record.get("case_name", ""),
                "paper_label": record.get("paper_label", ""),
                "run_type": record.get("run_type", ""),
                "provider": record.get("provider", ""),
                "model": record.get("model", ""),
                "best_single_test_accuracy": record.get("best_single_test_accuracy", ""),
                "best_single_test_macro_f1": record.get("best_single_test_macro_f1", ""),
                "ensemble_test_accuracy": record.get("ensemble_test_accuracy", ""),
                "ensemble_test_macro_f1": record.get("ensemble_test_macro_f1", ""),
                "final_choice": record.get("final_choice", ""),
            }
            for record in records
        ],
        output_path / "rm101_final_accuracy_table",
    )

    analysis_lines = ["# RM101 Multi-Model Analysis", ""]
    if records:
        best_by_f1 = max(
            records,
            key=lambda row: (
                float(row.get("best_single_test_macro_f1", 0) or 0),
                float(row.get("ensemble_test_macro_f1", 0) or 0),
                int(row.get("dag_depth", 0) or 0),
            ),
        )
        analysis_lines.extend(
            [
                f"- Runs discovered: {len(records)}",
                f"- Best case by test macro-F1: {best_by_f1.get('case_name', '')} / {best_by_f1.get('model', '')}",
                f"- Best choice: {best_by_f1.get('final_choice', '')}",
            ]
        )
    else:
        analysis_lines.append("- No runnable artifacts were found.")
    (output_path / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

    payload = {
        "root": str(root_path),
        "output_dir": str(output_path),
        "runs": records,
        "backend_comparison_csv": str(backend_csv),
        "backend_comparison_md": str(backend_md),
        "dag_summary_csv": str(dag_summary_csv),
        "dag_summary_md": str(dag_summary_md),
        "channel_alias_csv": str(alias_csv),
        "channel_alias_md": str(alias_md),
        "node_level_results_csv": str(node_csv),
        "node_level_results_md": str(node_md),
        "rm101_final_accuracy_csv": str(final_accuracy_csv),
        "rm101_final_accuracy_md": str(final_accuracy_md),
        "figures_dir": str(figures_dir),
        "analysis_path": str(output_path / "analysis.md"),
    }
    (output_path / "paper_bundle.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    if pd is not None and load_dotenv is not None and DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH)
    parser = argparse.ArgumentParser(description="Aggregate run artifacts into a paper-ready bundle.")
    parser.add_argument("--root", default="artifacts/rm101", help="Root directory containing model run folders.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/paper/rm101_multi_model",
        help="Directory for aggregated paper artifacts.",
    )
    args = parser.parse_args(argv)

    try:
        payload = export_paper_bundle(root=args.root, output_dir=args.output_dir)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
