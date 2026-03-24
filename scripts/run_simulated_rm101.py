from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_paper_phmga import export_paper_bundle
from scripts.run_full_dag_ml import run_full_dag_ml
from src.agents.shallow_ml_agent import BALANCED_MODEL_POOL
from src.config import load_case_config
from src.evaluation import compute_cross_dag_late_fusion
from src.manual_workflow import export_dag_artifacts
from src.simulated_rm101 import (
    SIMULATED_MODEL_TAGS,
    build_simulated_state,
    get_simulated_run_info,
    save_simulated_state,
    validate_complexity_ladder,
    write_simulated_summary,
)


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "rm101_simulated_v3"
DEFAULT_COMPARE_ROOT = ROOT / "artifacts" / "rm101_compare_v3"
DEFAULT_PAPER_OUTPUT = ROOT / "artifacts" / "paper" / "rm101_bigmodel_plus_gemini_simulated_v3"
BIGMODEL_BASELINE_DIR = ROOT / "artifacts" / "rm101" / "bigmodel__glm-4.7-flashx_paper_v1"


def _resolve_model_tags(model_tag: str) -> list[str]:
    if model_tag == "all":
        return list(SIMULATED_MODEL_TAGS)
    if model_tag not in SIMULATED_MODEL_TAGS:
        raise ValueError(f"Unsupported simulated model tag: {model_tag}")
    return [model_tag]


def _ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _write_markdown(path: str | Path, title: str, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No rows available._")
    else:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
            lines.append("| " + " | ".join(values) + " |")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _augment_manifest(run_dir: Path, *, run_type: str, paper_label: str) -> None:
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["run_type"] = run_type
    manifest["paper_label"] = paper_label
    _write_json(manifest_path, manifest)


def _copy_run_dir(source: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    return destination


def _default_parallel_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 8))


def _stage_bigmodel_baseline(compare_root: Path, case_config: dict[str, Any], *, parallel_workers: int) -> Path:
    if not BIGMODEL_BASELINE_DIR.exists():
        raise FileNotFoundError(f"Missing BigModel baseline directory: {BIGMODEL_BASELINE_DIR}")
    baseline_manifest = _read_json(BIGMODEL_BASELINE_DIR / "run_manifest.json")
    state_path = baseline_manifest.get("state_path")
    if not state_path:
        raise ValueError("BigModel baseline manifest is missing state_path.")
    run_dir = compare_root / "bigmodel__glm-4.7-flashx__real_baseline"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    payload = run_full_dag_ml(
        state_path=state_path,
        case_name=None,
        case_config=case_config,
        output_dir=run_dir,
        algorithm="balanced_pool",
        candidate_algorithms=list(BALANCED_MODEL_POOL),
        parallel_workers=parallel_workers,
        cv_folds=0,
        skip_llm_report=True,
    )
    _augment_manifest(run_dir, run_type="real_baseline", paper_label="BigModel / GLM-4.7-FlashX (real baseline)")
    return Path(payload["output_dir"])


def _build_and_run_simulated(
    model_tag: str,
    *,
    case_name: str,
    output_root: Path,
    parallel_workers: int,
    run_suffix: str = "simulated_paper_v3",
) -> Path:
    run_info = get_simulated_run_info(model_tag, run_suffix=run_suffix)
    run_dir = output_root / run_info.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    state, case_config = build_simulated_state(case_name=case_name, model_tag=model_tag)
    _ensure_dir(run_dir)
    state_path = run_dir / "builder_state.pkl"
    save_simulated_state(state, state_path)
    export_dag_artifacts(state, output_dir=run_dir / "graphs", stem="dag", max_nodes=None, save_png=True, save_json=True)
    write_simulated_summary(state, run_dir / "dag_summary.json")
    payload = run_full_dag_ml(
        state_path=state_path,
        case_name=None,
        case_config=case_config,
        output_dir=run_dir,
        algorithm="balanced_pool",
        candidate_algorithms=list(BALANCED_MODEL_POOL),
        parallel_workers=parallel_workers,
        cv_folds=0,
        skip_llm_report=True,
    )
    _augment_manifest(run_dir, run_type=run_info.run_type, paper_label=run_info.paper_label)
    return Path(payload["output_dir"])


def _stage_simulated_runs(run_dirs: Iterable[Path], compare_root: Path) -> list[Path]:
    staged_dirs: list[Path] = []
    for run_dir in run_dirs:
        target_name = run_dir.name
        for suffix in ("__simulated_paper_v3", "__simulated_mixed_paper_v1"):
            if target_name.endswith(suffix):
                target_name = target_name[: -len(suffix)] + "__simulated_variant"
                break
        staged_dirs.append(_copy_run_dir(run_dir, compare_root / target_name))
    return staged_dirs


def _write_cross_dag_outputs(paper_output_dir: Path, payload: dict[str, Any]) -> None:
    json_path = paper_output_dir / "cross_dag_late_fusion.json"
    csv_path = paper_output_dir / "cross_dag_late_fusion.csv"
    md_path = paper_output_dir / "cross_dag_late_fusion.md"
    _write_json(json_path, payload)
    rows = [
        {
            "selection_basis": payload.get("selection_basis", ""),
            "fusion_method": payload.get("fusion_method", ""),
            "n_val_windows": payload.get("n_val_windows", 0),
            "n_test_windows": payload.get("n_test_windows", 0),
            "val_accuracy": (payload.get("val_metrics") or {}).get("accuracy", 0.0),
            "val_macro_f1": (payload.get("val_metrics") or {}).get("macro_f1", 0.0),
            "test_accuracy": (payload.get("test_metrics") or {}).get("accuracy", 0.0),
            "test_macro_f1": (payload.get("test_metrics") or {}).get("macro_f1", 0.0),
        }
    ]
    _write_csv(csv_path, rows)
    weight_rows = list(payload.get("weights") or [])
    md_rows = rows + weight_rows
    _write_markdown(md_path, "Cross-DAG Late Fusion", md_rows)


def run_simulated_rm101(
    *,
    case_name: str = "case_exp2_paper",
    model_tag: str = "all",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    compare_root: str | Path = DEFAULT_COMPARE_ROOT,
    paper_output_dir: str | Path = DEFAULT_PAPER_OUTPUT,
    include_bigmodel_baseline: bool = False,
    parallel_workers: int | None = None,
    run_suffix: str = "simulated_paper_v3",
) -> dict[str, Any]:
    resolved_model_tags = _resolve_model_tags(model_tag)
    resolved_parallel_workers = int(parallel_workers or _default_parallel_workers())
    output_root_path = _ensure_dir(output_root)
    compare_root_path = Path(compare_root)
    paper_output_path = Path(paper_output_dir)
    if compare_root_path.exists():
        shutil.rmtree(compare_root_path)
    if paper_output_path.exists():
        shutil.rmtree(paper_output_path)
    compare_root_path = _ensure_dir(compare_root_path)
    paper_output_path = _ensure_dir(paper_output_path)

    case_config = load_case_config(case_name)
    simulated_run_dirs = [
        _build_and_run_simulated(
            tag,
            case_name=case_name,
            output_root=output_root_path,
            parallel_workers=resolved_parallel_workers,
            run_suffix=run_suffix,
        )
        for tag in resolved_model_tags
    ]
    if len(resolved_model_tags) == len(SIMULATED_MODEL_TAGS):
        states_for_validation = {}
        for tag, run_dir in zip(resolved_model_tags, simulated_run_dirs):
            import pickle

            with (run_dir / "builder_state.pkl").open("rb") as handle:
                states_for_validation[tag] = pickle.load(handle)
        ladder_summary = validate_complexity_ladder(states_for_validation)
    else:
        ladder_summary = {
            tag: _read_json(run_dir / "dag_summary.json")
            for tag, run_dir in zip(resolved_model_tags, simulated_run_dirs)
        }

    staged_run_dirs = _stage_simulated_runs(simulated_run_dirs, compare_root_path)
    if include_bigmodel_baseline:
        staged_run_dirs.insert(0, _stage_bigmodel_baseline(compare_root_path, case_config, parallel_workers=resolved_parallel_workers))

    paper_bundle = export_paper_bundle(root=compare_root_path, output_dir=paper_output_path)
    late_fusion = compute_cross_dag_late_fusion(staged_run_dirs)
    _write_cross_dag_outputs(paper_output_path, late_fusion)

    payload = {
        "case_name": case_name,
        "model_tags": resolved_model_tags,
        "output_root": str(output_root_path),
        "compare_root": str(compare_root_path),
        "paper_output_dir": str(paper_output_path),
        "simulated_run_dirs": [str(path) for path in simulated_run_dirs],
        "staged_run_dirs": [str(path) for path in staged_run_dirs],
        "include_bigmodel_baseline": include_bigmodel_baseline,
        "parallel_workers": resolved_parallel_workers,
        "complexity_ladder": ladder_summary,
        "paper_bundle": paper_bundle,
        "cross_dag_late_fusion": late_fusion,
    }
    _write_json(paper_output_path / "simulated_rm101_bundle.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline simulated RM101 Gemini planner variants.")
    parser.add_argument("--case", default="case_exp2_paper", help="Case name under config/.")
    parser.add_argument(
        "--model-tag",
        default="all",
        choices=["all", *SIMULATED_MODEL_TAGS],
        help="Single simulated Gemini model tag or all variants.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for simulated run outputs.")
    parser.add_argument("--compare-root", default=str(DEFAULT_COMPARE_ROOT), help="Directory staged for exporter scans.")
    parser.add_argument("--paper-output-dir", default=str(DEFAULT_PAPER_OUTPUT), help="Directory for paper artifacts.")
    parser.add_argument("--include-bigmodel-baseline", action="store_true", help="Include the real BigModel baseline in the compare bundle.")
    parser.add_argument("--parallel-workers", type=int, default=0, help="Parallel workers for leaf x model evaluation.")
    parser.add_argument("--run-suffix", default="simulated_paper_v3", help="Suffix used for per-model simulated run directories.")
    args = parser.parse_args(argv)

    try:
        payload = run_simulated_rm101(
            case_name=args.case,
            model_tag=args.model_tag,
            output_root=args.output_root,
            compare_root=args.compare_root,
            paper_output_dir=args.paper_output_dir,
            include_bigmodel_baseline=args.include_bigmodel_baseline,
            parallel_workers=args.parallel_workers or None,
            run_suffix=args.run_suffix,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
