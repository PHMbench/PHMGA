"""Export the current experiment ledger into a self-contained paper_phmga bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, write_json, write_text


LEDGER_PATH = ROOT / "doc/experiments/01_result_ledger.md"
WORKER_RESULTS_DIR = ROOT / "doc/experiments/handoff/results"
ARTIFACTS_ROOT = ROOT / "artifacts/paper"
DEFAULT_OUTPUT_DIR = ROOT / "paper_phmga"
TEXT_ARTIFACT_SUFFIXES = {".json", ".md", ".txt"}
KNOWN_ARTIFACTS = [
    "dag.json",
    "validated_dag.json",
    "compiled_dag_manifest.json",
    "dag_graph.md",
    "resolved_splits.json",
    "resolved_dataset_manifest.json",
    "resolved_config.json",
    "workflow_state.json",
    "artifact_index.json",
    "dag_quality_summary.json",
    "dataset_level_runtime_trace.json",
    "decision_side_outputs.json",
    "feature_pipeline.json",
    "feature_list.json",
    "feature_separability_summary.json",
    "metrics.json",
    "predictions.json",
    "importance.json",
    "similarity_artifacts.json",
    "model_build_plan.json",
    "training_curves.json",
    "checkpoint.json",
    "control_statistics.json",
    "planner_transport_trace.json",
    "planner_normalization_trace.json",
    "planner_raw_response.txt",
    "planner_repair_response.txt",
    "step_plan.json",
    "final_report.md",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_yaml_block(text: str) -> Dict[str, Any]:
    match = re.search(r"```yaml\n(.*?)\n```", text, re.DOTALL)
    if not match:
        return {}
    return yaml.safe_load(match.group(1)) or {}


def _coerce_cell(value: str) -> Any:
    cleaned = value.strip().strip("`").strip()
    if cleaned == "":
        return None
    lowered = cleaned.lower()
    if lowered in {"accept", "reject", "pending", "pass", "fail", "yes", "no", "n/a"}:
        return lowered if lowered != "n/a" else "n/a"
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    return cleaned


def _parse_markdown_table(lines: Iterable[str]) -> List[Dict[str, Any]]:
    table_lines = [line.rstrip() for line in lines if line.strip()]
    if len(table_lines) < 2:
        return []
    headers = [cell.strip() for cell in table_lines[0].strip().strip("|").split("|")]
    rows: List[Dict[str, Any]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append({header: _coerce_cell(cell) for header, cell in zip(headers, cells)})
    return rows


def parse_ledger(path: Path = LEDGER_PATH) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    yaml_block = _extract_yaml_block(text)
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("| experiment_id |"))
    table_lines: List[str] = []
    for line in lines[header_idx:]:
        if not line.strip().startswith("|"):
            break
        table_lines.append(line)
    rows = _parse_markdown_table(table_lines)
    return yaml_block, rows


def _parse_key_value_bullets(lines: Iterable[str]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        key, _, value = stripped[2:].partition(":")
        payload[key.strip()] = value.strip()
    return payload


def parse_worker_result(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    top_lines: List[str] = []
    idx = 0
    for idx, line in enumerate(lines):
        if line.startswith("## "):
            break
        top_lines.append(line)
    else:
        idx = len(lines)

    metadata = _parse_key_value_bullets(top_lines)
    sections: Dict[str, str] = {}
    subsections: Dict[str, Dict[str, str]] = {}
    current_h2: Optional[str] = None
    current_h3: Optional[str] = None
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer, current_h2, current_h3
        content = "\n".join(buffer).strip()
        if current_h2 and current_h3:
            subsections.setdefault(current_h2, {})[current_h3] = content
        elif current_h2:
            sections[current_h2] = content
        buffer = []

    for line in lines[idx:]:
        if line.startswith("## "):
            flush()
            current_h2 = line[3:].strip()
            current_h3 = None
            continue
        if line.startswith("### "):
            flush()
            current_h3 = line[4:].strip()
            continue
        buffer.append(line)
    flush()

    artifact_checklist = _parse_key_value_bullets(sections.get("Artifact Checklist", "").splitlines())
    required_evidence = subsections.get("Required Evidence", {})

    return {
        "metadata": metadata,
        "artifact_checklist": artifact_checklist,
        "required_evidence": required_evidence,
        "metrics_summary": sections.get("Metrics Summary", "").strip(),
        "failure_summary": sections.get("Failure Summary", "").strip(),
        "notes": sections.get("Notes", "").strip(),
        "raw_markdown": text,
        "path": str(path.relative_to(ROOT)),
    }


def _stage_name_from_row(row: Dict[str, Any]) -> str:
    note = str(row.get("note") or "")
    experiment_id = str(row["experiment_id"])
    if experiment_id.endswith("_pilot_v1") or "pilot smoke" in note:
        return "stage_a_pilot"
    if "backend comparison" in note or "historical comparison failure" in note:
        return "stage_b_backend_comparison"
    if "formal main" in note or "path comparison" in note:
        return "stage_c_formal_main"
    return "stage_d_ablation"


def _run_type_from_row(row: Dict[str, Any]) -> str:
    stage_name = _stage_name_from_row(row)
    if stage_name == "stage_a_pilot":
        return "pilot"
    if stage_name == "stage_b_backend_comparison":
        return "backend_comparison"
    if stage_name == "stage_c_formal_main":
        return "main"
    return "ablation"


def _status_from_row(row: Dict[str, Any]) -> str:
    keep = row.get("keep")
    if keep in {"accept", "reject"}:
        return str(keep)
    return "pending"


def _parse_provider_model(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = re.search(r"([A-Za-z0-9_.-]+)\s*/\s*([^;\n]+)", text)
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _derive_provider_model(
    row: Dict[str, Any],
    worker_result: Optional[Dict[str, Any]],
    ledger_meta: Dict[str, Any],
) -> Tuple[str, str]:
    if worker_result:
        provider_model = worker_result["metadata"].get("provider/model", "")
        provider, model = _parse_provider_model(provider_model)
        if provider and model:
            return provider, model

    llm_mode = str(row.get("llm_mode") or "")
    if llm_mode == "offline_stub":
        return "offline_stub", "n/a"

    note = str(row.get("note") or "")
    provider, model = _parse_provider_model(note)
    if provider and model:
        return provider, model

    stage_name = _stage_name_from_row(row)
    selected = ledger_meta.get("selected_global_best_backend", {}) or {}
    if stage_name in {"stage_c_formal_main", "stage_d_ablation"} and selected.get("selected_from_stage_b"):
        return str(selected.get("provider") or "pending"), str(selected.get("model") or "pending")

    return "pending", "pending"


def _output_dir_for_row(row: Dict[str, Any]) -> Optional[Path]:
    raw = row.get("output_dir")
    if raw in {None, "", "n/a"}:
        return None
    candidate = ROOT / str(raw)
    return candidate


def _copy_worker_result(worker_result: Optional[Dict[str, Any]], dest_dir: Path) -> Optional[str]:
    if not worker_result:
        return None
    target = dest_dir / "worker_result.md"
    write_text(worker_result["raw_markdown"], target)
    return str(target.relative_to(dest_dir))


def _copy_artifacts(row: Dict[str, Any], dest_dir: Path) -> Dict[str, Any]:
    source_dir = _output_dir_for_row(row)
    artifacts_dest = ensure_dir(dest_dir / "artifacts")
    copied: List[str] = []
    presence = {name: False for name in KNOWN_ARTIFACTS}
    validated_dag_alias = None

    if source_dir and source_dir.exists():
        for source_path in sorted(source_dir.iterdir()):
            if not source_path.is_file() or source_path.suffix not in TEXT_ARTIFACT_SUFFIXES:
                continue
            target_path = artifacts_dest / source_path.name
            shutil.copy2(source_path, target_path)
            copied.append(source_path.name)
            presence[source_path.name] = True

    if presence.get("dag.json") and not presence.get("validated_dag.json"):
        validated_dag_alias = "dag.json"

    artifact_manifest = {
        "source_output_dir": str(source_dir.relative_to(ROOT)) if source_dir else None,
        "source_output_exists": bool(source_dir and source_dir.exists()),
        "copied_artifacts": copied,
        "artifact_presence": presence,
        "validated_dag_alias": validated_dag_alias,
    }
    write_json(artifact_manifest, dest_dir / "artifact_manifest.json")
    return artifact_manifest


def _load_json(path: Optional[Path]) -> Optional[Any]:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(artifact_manifest: Dict[str, Any], evidence_dir: Path, filename: str) -> Optional[Path]:
    if filename in artifact_manifest["copied_artifacts"]:
        return evidence_dir / "artifacts" / filename
    return None


def _format_metric_line(metrics: Optional[Dict[str, Any]], ledger_value: Any, key: str) -> str:
    if isinstance(metrics, dict):
        test_metrics = metrics.get("test")
        if isinstance(test_metrics, dict) and key in test_metrics:
            return f"- test_{key}: {test_metrics[key]}"
    if ledger_value is not None:
        return f"- ledger_{key}: {ledger_value}"
    return f"- {key}: unavailable"


def _front_matter(payload: Dict[str, Any]) -> str:
    return "---\n" + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip() + "\n---"


def _build_subdoc(
    row: Dict[str, Any],
    ledger_meta: Dict[str, Any],
    worker_result: Optional[Dict[str, Any]],
    artifact_manifest: Dict[str, Any],
    evidence_dir: Path,
    bundle_evidence_dir: str,
) -> str:
    provider, model = _derive_provider_model(row, worker_result, ledger_meta)
    status = _status_from_row(row)
    experiment_id = str(row["experiment_id"])
    stage_name = _stage_name_from_row(row)
    metrics = _load_json(_artifact_path(artifact_manifest, evidence_dir, "metrics.json"))
    feature_summary = _load_json(_artifact_path(artifact_manifest, evidence_dir, "feature_separability_summary.json"))
    worker_metadata = worker_result["metadata"] if worker_result else {}
    required_evidence = worker_result["required_evidence"] if worker_result else {}

    front_matter = {
        "experiment_id": experiment_id,
        "dataset": row.get("dataset"),
        "graph_path": row.get("graph_path"),
        "phase": row.get("phase"),
        "run_type": _run_type_from_row(row),
        "provider": provider,
        "model": model,
        "status": status,
        "artifact_contract_pass": row.get("artifact_contract_pass"),
        "feature_separability_pass": row.get("feature_separability_pass"),
        "selection_eligible": row.get("selection_eligible"),
        "output_dir": row.get("output_dir"),
        "bundle_evidence_dir": bundle_evidence_dir,
    }

    summary = (
        f"`{experiment_id}` is a `{_run_type_from_row(row)}` run on `{row.get('dataset')}` / `{row.get('graph_path')}` "
        f"with backend `{provider} / {model}`. Canonical status is `{status}` from the result ledger."
    )

    status_lines = [
        f"- stage: `{stage_name}`",
        f"- phase: `{row.get('phase')}`",
        f"- artifact_contract_pass: `{row.get('artifact_contract_pass')}`",
        f"- feature_separability_pass: `{row.get('feature_separability_pass')}`",
        f"- selection_eligible: `{row.get('selection_eligible')}`",
        f"- worker_result_present: `{'yes' if worker_result else 'no'}`",
    ]

    metrics_lines = [
        _format_metric_line(metrics, row.get("accuracy"), "accuracy"),
        _format_metric_line(metrics, row.get("macro_f1"), "macro_f1"),
    ]

    artifact_lines = [
        f"- source_output_dir: `{artifact_manifest['source_output_dir']}`",
        f"- copied_artifact_count: `{len(artifact_manifest['copied_artifacts'])}`",
        f"- artifact_manifest: `artifact_manifest.json`",
        f"- ledger_row: `ledger_row.json`",
    ]
    if worker_result:
        artifact_lines.append("- worker_result: `worker_result.md`")
    if artifact_manifest.get("validated_dag_alias"):
        artifact_lines.append(f"- validated_dag_alias: `{artifact_manifest['validated_dag_alias']}`")

    evidence_lines: List[str] = []
    if isinstance(feature_summary, dict):
        decision = feature_summary.get("decision", "unknown")
        evidence_lines.append(f"- feature_summary_decision: `{decision}`")
        aggregate_scores = feature_summary.get("aggregate_scores", {}) or {}
        for key in ("mean_fisher_score", "median_fisher_score", "top5_mean_score"):
            if key in aggregate_scores:
                evidence_lines.append(f"- {key}: {aggregate_scores[key]}")
        split_stability = feature_summary.get("split_stability", {}) or {}
        if "train_val_rank_corr" in split_stability:
            evidence_lines.append(f"- train_val_rank_corr: {split_stability['train_val_rank_corr']}")
        top_features = feature_summary.get("top_features", []) or []
        if top_features:
            rendered_top = ", ".join(
                f"{item.get('name')} ({item.get('score')})" for item in top_features[:5] if isinstance(item, dict)
            )
            evidence_lines.append(f"- top_features: {rendered_top}")
    else:
        feature_text = required_evidence.get("feature_separability_summary", "").strip()
        if feature_text:
            evidence_lines.append(feature_text)
        else:
            evidence_lines.append("No runtime-native feature separability evidence is available in this bundle.")

    failure_lines: List[str] = []
    if worker_result and worker_result.get("failure_summary"):
        failure_lines.append(worker_result["failure_summary"])
    elif worker_result and required_evidence.get("progress_record"):
        failure_lines.append(required_evidence["progress_record"].strip())
    note = str(row.get("note") or "").strip()
    if note:
        failure_lines.append(f"Ledger note: {note}")
    if not failure_lines:
        failure_lines.append("No failure note was recorded. This row is currently pending or accepted without extra narrative.")

    sections = [
        _front_matter(front_matter),
        "",
        "# Summary",
        "",
        summary,
        "",
        "# Status",
        "",
        "\n".join(status_lines),
        "",
        "# Metrics",
        "",
        "\n".join(metrics_lines),
        "",
        "# Artifacts",
        "",
        "\n".join(artifact_lines),
        "",
        "# Feature / Diagnosis Evidence",
        "",
        "\n".join(evidence_lines),
        "",
        "# Failure Or Pending Notes",
        "",
        "\n\n".join(failure_lines),
        "",
    ]
    return "\n".join(sections)


def _build_metadata(
    row: Dict[str, Any],
    ledger_meta: Dict[str, Any],
    worker_result: Optional[Dict[str, Any]],
    artifact_manifest: Dict[str, Any],
) -> Dict[str, Any]:
    provider, model = _derive_provider_model(row, worker_result, ledger_meta)
    worker_artifacts = worker_result.get("artifact_checklist", {}) if worker_result else {}
    mismatch = False
    for artifact_name, worker_value in worker_artifacts.items():
        actual = artifact_manifest["artifact_presence"].get(artifact_name)
        if actual is not None and worker_value in {"yes", "no"} and actual != (worker_value == "yes"):
            mismatch = True
            break
    return {
        "experiment_id": row["experiment_id"],
        "ledger_row": row,
        "worker_result": worker_result,
        "copied_artifacts": artifact_manifest["copied_artifacts"],
        "artifact_presence_map": artifact_manifest["artifact_presence"],
        "derived_stage_name": _stage_name_from_row(row),
        "derived_run_type": _run_type_from_row(row),
        "derived_backend_tuple": {"provider": provider, "model": model},
        "artifact_mismatch": mismatch,
        "validated_dag_alias": artifact_manifest.get("validated_dag_alias"),
        "source_provenance": {
            "ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
            "worker_results_dir": str(WORKER_RESULTS_DIR.relative_to(ROOT)),
            "artifacts_root": str(ARTIFACTS_ROOT.relative_to(ROOT)),
        },
    }


def _build_stage_doc(
    stage_name: str,
    rows: List[Dict[str, Any]],
    selection_status: Dict[str, Any],
) -> str:
    title_map = {
        "stage_a_pilot": "Stage A Pilot",
        "stage_b_backend_comparison": "Stage B Backend Comparison",
        "stage_c_formal_main": "Stage C Formal Main",
        "stage_d_ablation": "Stage D Ablation",
    }
    lines = [f"# {title_map[stage_name]}", ""]
    if stage_name == "stage_b_backend_comparison":
        lines.extend(
            [
                "Current active Stage B set and selection state are sourced from `selection_status.json`.",
                "",
                f"- active_codex: `{selection_status['active_stage_b_set']['codex']['provider']} / {selection_status['active_stage_b_set']['codex']['model']}`",
                f"- active_openrouter: `{selection_status['active_stage_b_set']['openrouter']['provider']} / {selection_status['active_stage_b_set']['openrouter']['model']}`",
                f"- selected_backend_status: `{selection_status['selected_global_best_backend']['status']}`",
                "",
            ]
        )
        historical = selection_status.get("historical_failure_not_in_active_set", [])
        if historical:
            lines.append(f"- historical_failure_not_in_active_set: {', '.join(f'`{item}`' for item in historical)}")
            lines.append("")
    lines.extend(
        [
            "| experiment_id | dataset | graph_path | status | subdoc |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        experiment_id = str(row["experiment_id"])
        lines.append(
            f"| {experiment_id} | {row.get('dataset')} | {row.get('graph_path')} | {_status_from_row(row)} | [subdoc](../subdocs/{experiment_id}.md) |"
        )
    lines.append("")
    return "\n".join(lines)


def export_paper_phmga(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    ledger_meta, rows = parse_ledger()
    worker_results = {
        path.stem: parse_worker_result(path)
        for path in sorted(WORKER_RESULTS_DIR.glob("*.md"))
        if path.name.lower() != "readme.md"
    }

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    subdocs_dir = ensure_dir(output_dir / "subdocs")
    evidence_root = ensure_dir(output_dir / "evidence")
    stages_dir = ensure_dir(output_dir / "stages")

    experiment_index: List[Dict[str, Any]] = []
    stage_index: Dict[str, List[str]] = {
        "stage_a_pilot": [],
        "stage_b_backend_comparison": [],
        "stage_c_formal_main": [],
        "stage_d_ablation": [],
    }

    for row in rows:
        experiment_id = str(row["experiment_id"])
        stage_name = _stage_name_from_row(row)
        stage_index[stage_name].append(experiment_id)

        evidence_dir = ensure_dir(evidence_root / experiment_id)
        worker_result = worker_results.get(experiment_id)
        artifact_manifest = _copy_artifacts(row, evidence_dir)
        write_json(row, evidence_dir / "ledger_row.json")
        _copy_worker_result(worker_result, evidence_dir)
        metadata = _build_metadata(row, ledger_meta, worker_result, artifact_manifest)
        write_json(metadata, evidence_dir / "metadata.json")

        subdoc = _build_subdoc(
            row,
            ledger_meta,
            worker_result,
            artifact_manifest,
            evidence_dir,
            f"evidence/{experiment_id}",
        )
        write_text(subdoc, subdocs_dir / f"{experiment_id}.md")

        experiment_index.append(
            {
                "experiment_id": experiment_id,
                "dataset": row.get("dataset"),
                "graph_path": row.get("graph_path"),
                "stage_name": stage_name,
                "status": _status_from_row(row),
                "subdoc_path": f"subdocs/{experiment_id}.md",
                "evidence_dir": f"evidence/{experiment_id}",
                "provider": metadata["derived_backend_tuple"]["provider"],
                "model": metadata["derived_backend_tuple"]["model"],
            }
        )

    historical_failure_not_in_active_set = [
        str(row["experiment_id"])
        for row in rows
        if _stage_name_from_row(row) == "stage_b_backend_comparison"
        and "historical comparison failure" in str(row.get("note") or "")
    ]

    selection_status = {
        "active_stage_b_set": ledger_meta.get("active_stage_b_set", {}),
        "selected_global_best_backend": ledger_meta.get("selected_global_best_backend", {}),
        "historical_failure_not_in_active_set": historical_failure_not_in_active_set,
    }
    write_json(selection_status, output_dir / "selection_status.json")

    for stage_name, experiment_ids in stage_index.items():
        stage_rows = [row for row in rows if str(row["experiment_id"]) in experiment_ids]
        write_text(_build_stage_doc(stage_name, stage_rows, selection_status), stages_dir / f"{stage_name}.md")

    stage_counts = {key: len(value) for key, value in stage_index.items()}
    manifest = {
        "generated_at": _utc_now(),
        "source_ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
        "experiment_count": len(rows),
        "stage_counts": stage_counts,
        "selected_global_best_backend": ledger_meta.get("selected_global_best_backend", {}),
        "bundle_mode": "self_contained_snapshot",
    }
    write_json(manifest, output_dir / "manifest.json")
    write_json(experiment_index, output_dir / "experiment_index.json")
    write_json(stage_index, output_dir / "stage_index.json")

    readme_lines = [
        "# paper_phmga",
        "",
        "This directory is a self-contained experiment snapshot bundle for autoresearch consumption.",
        "",
        f"- generated_at: `{manifest['generated_at']}`",
        f"- source_ledger: `{manifest['source_ledger_path']}`",
        f"- experiment_count: `{manifest['experiment_count']}`",
        f"- selected_backend_status: `{manifest['selected_global_best_backend'].get('status', 'pending')}`",
        "",
        "Primary entrypoints:",
        "",
        "- `subdocs/*.md`: one child document per `experiment_id`",
        "- `selection_status.json`: current Stage B active set and winner state",
        "- `manifest.json`: bundle-level metadata",
        "- `evidence/<experiment_id>/`: self-contained evidence package",
        "",
    ]
    write_text("\n".join(readme_lines), output_dir / "README.md")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    manifest = export_paper_phmga(Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
