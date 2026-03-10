#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import re


def _resolve_latest_run_dir(case_dir: Path) -> Path | None:
    if not case_dir.exists():
        return None
    candidates = [p for p in case_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    ts_pattern = re.compile(r"^\d{8}-\d{6}$")

    def _rank(path: Path) -> tuple[int, float, str]:
        name = path.name
        if ts_pattern.match(name):
            score = 2
        elif name.startswith("run-"):
            score = 1
        else:
            score = 0
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = 0.0
        return (score, mtime, name)

    return sorted(candidates, key=_rank)[-1]


def _path_or_empty(path: Path) -> str:
    return str(path) if path.exists() else ""


def _resolve_metrics_path(case_dir: Path, run_dir: Path | None) -> str:
    candidates: list[Path] = []
    if run_dir is not None:
        candidates.append(run_dir / "metrics.json")
        candidates.append(run_dir / "logs" / "metrics.json")
    candidates.extend(case_dir.glob("**/metrics.json"))

    existing: list[Path] = [p for p in candidates if p.exists() and p.is_file()]
    if existing:
        existing.sort(key=lambda p: (float(p.stat().st_mtime), str(p)))
        return str(existing[-1])

    # Shallow route may not emit metrics.json; create a fallback file so evidence chain is closed.
    fallback_path = case_dir / "metrics.fallback.json"
    fallback_payload = {
        "is_fallback": True,
        "status": "missing_metrics_artifact",
        "reason": "No metrics.json discovered under case_dir.",
        "run_dir": str(run_dir) if run_dir is not None else "",
        "val_acc": 0.0,
        "val_macro_f1": 0.0,
        "test_acc": 0.0,
        "test_macro_f1": 0.0,
    }
    fallback_path.write_text(json.dumps(fallback_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(fallback_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover latest run artifacts under a case directory.")
    parser.add_argument("--case-dir", required=True, help="Case output directory.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    run_dir = _resolve_latest_run_dir(case_dir)

    payload: Dict[str, str] = {
        "run_dir": "",
        "metrics_path": "",
        "dataset_manifest_path": "",
        "config_resolve_path": "",
        "model_config_resolved_path": "",
        "preflight_report_path": "",
        "compatibility_report_path": "",
        "contract_violation_report_path": "",
        "dag_compile_report_path": "",
        "predictions_path": "",
        "operator_importance_path": "",
        "final_report_path": "",
    }
    if run_dir is not None:
        payload["run_dir"] = str(run_dir)
        payload["metrics_path"] = _resolve_metrics_path(case_dir, run_dir)
        payload["dataset_manifest_path"] = _path_or_empty(run_dir / "dataset_manifest.json")
        payload["config_resolve_path"] = _path_or_empty(run_dir / "config_resolve.json")
        payload["model_config_resolved_path"] = _path_or_empty(run_dir / "model_config.resolved.yaml")
        payload["preflight_report_path"] = _path_or_empty(run_dir / "preflight_report.json")
        payload["compatibility_report_path"] = _path_or_empty(run_dir / "compatibility_report.json")
        payload["contract_violation_report_path"] = _path_or_empty(run_dir / "contract_violation_report.json")
        payload["dag_compile_report_path"] = _path_or_empty(run_dir / "dag_compile_report.json")
        payload["predictions_path"] = _path_or_empty(run_dir / "predictions.csv")
        payload["operator_importance_path"] = _path_or_empty(run_dir / "explain" / "operator_importance.json")
    elif case_dir.exists():
        payload["metrics_path"] = _resolve_metrics_path(case_dir, None)
    payload["final_report_path"] = _path_or_empty(case_dir / "final_report.md")

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
