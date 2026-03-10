#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _metric(record: Dict[str, Any], key: str) -> float | None:
    metrics_path = Path(str(record.get("metrics_path") or ""))
    if not metrics_path.exists():
        return None
    metrics = _read_json(metrics_path)
    if key in metrics:
        try:
            return float(metrics[key])
        except Exception:
            return None
    val_obj = metrics.get("val")
    if isinstance(val_obj, dict) and key in val_obj:
        try:
            return float(val_obj[key])
        except Exception:
            return None
    return None


def _to_md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _safe_fmt(v: float | None, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-oriented analysis draft from matrix manifest.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = list(_iter_manifest(manifest_path))
    if not records:
        raise SystemExit(f"No manifest rows found: {manifest_path}")

    success = [r for r in records if str(r.get("status")) == "ok"]
    failed = [r for r in records if str(r.get("status")) != "ok"]

    # Main table
    main_rows: List[List[str]] = []
    for rec in records:
        main_rows.append(
            [
                str(rec.get("dataset", "")),
                str(rec.get("llm", "")),
                str(rec.get("ablation", "")),
                str(rec.get("status", "")),
                _safe_fmt(_metric(rec, "val_acc")),
                _safe_fmt(_metric(rec, "val_macro_f1")),
                str(rec.get("run_dir", "")),
            ]
        )

    # Ablation delta table: A0 - A1, A0 - A2 by (dataset,llm)
    by_key: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for rec in success:
        key = (str(rec.get("dataset", "")), str(rec.get("llm", "")))
        by_key[key][str(rec.get("ablation", ""))] = rec

    delta_rows: List[List[str]] = []
    for (dataset, llm), group in sorted(by_key.items()):
        a0 = group.get("A0_full")
        a1 = group.get("A1_no_reflect")
        a2 = group.get("A2_no_prior")
        if not a0:
            continue
        a0_acc = _metric(a0, "val_acc")
        a1_acc = _metric(a1, "val_acc") if a1 else None
        a2_acc = _metric(a2, "val_acc") if a2 else None
        d01 = (a0_acc - a1_acc) if (a0_acc is not None and a1_acc is not None) else None
        d02 = (a0_acc - a2_acc) if (a0_acc is not None and a2_acc is not None) else None
        delta_rows.append([dataset, llm, _safe_fmt(a0_acc), _safe_fmt(a1_acc), _safe_fmt(d01), _safe_fmt(a2_acc), _safe_fmt(d02)])

    # Dataset-level mean for A0
    dataset_mean_rows: List[List[str]] = []
    per_dataset_vals: Dict[str, List[float]] = defaultdict(list)
    for rec in success:
        if str(rec.get("ablation")) != "A0_full":
            continue
        acc = _metric(rec, "val_acc")
        if acc is not None:
            per_dataset_vals[str(rec.get("dataset", ""))].append(acc)
    for dataset, vals in sorted(per_dataset_vals.items()):
        mean_v = sum(vals) / len(vals) if vals else 0.0
        dataset_mean_rows.append([dataset, str(len(vals)), _safe_fmt(mean_v)])

    # Operator evidence presence
    evidence_present = 0
    for rec in success:
        run_dir = Path(str(rec.get("run_dir") or ""))
        if (run_dir / "explain" / "operator_importance.json").exists():
            evidence_present += 1

    # Draft conclusion hints
    bullets: List[str] = []
    if failed:
        bullets.append(f"- 共有 `{len(failed)}` 个组合失败，优先基于 `preflight_log/run_log` 做失败归因。")
    else:
        bullets.append("- 当前清单下所有组合均成功运行。")
    if delta_rows:
        pos_reflect = sum(1 for r in delta_rows if r[4] and float(r[4]) > 0)
        pos_prior = sum(1 for r in delta_rows if r[6] and float(r[6]) > 0)
        bullets.append(f"- `A0-A1` 在 `{pos_reflect}/{len(delta_rows)}` 个 (dataset,llm) 组合上为正增益。")
        bullets.append(f"- `A0-A2` 在 `{pos_prior}/{len(delta_rows)}` 个 (dataset,llm) 组合上为正增益。")
    bullets.append(f"- 可解释性证据文件（`operator_importance.json`）覆盖 `{evidence_present}/{len(success)}` 个成功组合。")

    md = []
    md.append("# PHMGA Matrix Analysis Draft")
    md.append("")
    md.append("## 1) Main Results")
    md.append("")
    md.append(_to_md_table(
        ["dataset", "llm", "ablation", "status", "val_acc", "val_macro_f1", "run_dir"],
        main_rows,
    ))
    md.append("")
    md.append("## 2) Ablation Gains (A0 baseline)")
    md.append("")
    if delta_rows:
        md.append(_to_md_table(
            ["dataset", "llm", "A0_acc", "A1_acc", "A0-A1", "A2_acc", "A0-A2"],
            delta_rows,
        ))
    else:
        md.append("_No complete A0/A1/A2 groups found._")
    md.append("")
    md.append("## 3) Dataset-level Generalization View (A0)")
    md.append("")
    if dataset_mean_rows:
        md.append(_to_md_table(["dataset", "n_runs", "mean_val_acc"], dataset_mean_rows))
    else:
        md.append("_No A0 successful runs found._")
    md.append("")
    md.append("## 4) Explainability Evidence Coverage")
    md.append("")
    md.append(f"- successful runs: `{len(success)}`")
    md.append(f"- runs with `explain/operator_importance.json`: `{evidence_present}`")
    md.append("")
    md.append("## 5) Draft Conclusions")
    md.append("")
    md.extend(bullets)
    md.append("")
    md.append("## 6) Reproducibility Checklist")
    md.append("")
    md.append("- 固定 manifest 输入：`manifest.jsonl`")
    md.append("- 核对配置：`resolved_case_config` 与 `config_resolve.json`")
    md.append("- 核对指标来源：`metrics.json`")
    md.append("- 核对可解释性：`operator_importance.json`")

    draft_path = output_dir / "analysis_draft.md"
    draft_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[analysis] wrote {draft_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
