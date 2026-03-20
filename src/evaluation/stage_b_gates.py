"""Executable gate rules for Stage B backend comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict


REQUIRED_STAGE_B_ARTIFACTS = (
    "validated_dag.json",
    "compiled_dag_manifest.json",
    "feature_pipeline.json",
    "feature_list.json",
    "feature_separability_summary.json",
    "artifact_index.json",
    "metrics.json",
    "final_report.md",
)


def evaluate_artifact_contract(output_dir: str | Path) -> bool:
    """Return whether the Stage B hard-gate artifact set exists."""

    root = Path(output_dir)
    return all((root / artifact_name).exists() for artifact_name in REQUIRED_STAGE_B_ARTIFACTS)


def evaluate_feature_separability(summary: Dict[str, Any]) -> bool:
    """Return whether the minimal separability gate passes."""

    feature_count = int(summary.get("feature_count", 0) or 0)
    non_empty_feature_count = int(summary.get("non_empty_feature_count", 0) or 0)
    constant_feature_count = int(summary.get("constant_feature_count", 0) or 0)
    top_features = summary.get("top_features", [])
    aggregate_scores = dict(summary.get("aggregate_scores", {}))
    split_stability = dict(summary.get("split_stability", {}))
    top5_mean_score = float(aggregate_scores.get("top5_mean_score", 0.0) or 0.0)
    rank_corr = split_stability.get("train_val_rank_corr")
    decision = str(summary.get("decision", "")).strip().lower()

    if feature_count <= 0:
        return False
    if non_empty_feature_count <= 0:
        return False
    if constant_feature_count >= feature_count:
        return False
    if not isinstance(top_features, list) or not top_features:
        return False
    if top5_mean_score <= 0.0:
        return False
    if not isinstance(rank_corr, (int, float)) or not math.isfinite(float(rank_corr)):
        return False
    return decision == "pass"


def evaluate_selection_eligibility(*, keep: str, artifact_contract_pass: bool, feature_separability_pass: bool) -> bool:
    """Return whether a Stage B row may enter backend selection."""

    return keep == "accept" and artifact_contract_pass and feature_separability_pass
