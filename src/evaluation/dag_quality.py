"""Compact DAG quality evaluation for the paper-oriented workflow.

This module intentionally stays small. It does not try to become a second
training system or a platform-wide scoreboard. Its job is to summarize the
current round into a few signals that help `reflect_agent` decide whether the
DAG should finish, patch, or replan.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.bridge import compile_dag_for_path
from src.data import DatasetProtocol, build_dataset_views, materialize_proxy_split_signals
from src.model import run_shallow_ml_baseline
from src.operators import OperatorCatalog
from src.states import WorkflowState


RecommendationHint = Literal["finish_candidate", "patch_candidate", "replan_candidate", "halt_candidate"]


class DagQualitySummary(BaseModel):
    """Compact current-round quality summary for reflection and reporting."""

    current_depth: int
    min_depth: int
    max_depth: int
    depth_ok: bool

    feature_node_count: int
    multi_node_count: int
    operator_categories: List[str] = Field(default_factory=list)

    execution_gap_count: int
    nan_ratio: float
    zero_variance_ratio: float

    proxy_probe_enabled: bool
    proxy_probe_macro_f1: Optional[float] = None

    issues: List[str] = Field(default_factory=list)
    recommendation_hint: RecommendationHint


def _dag_depth(state: WorkflowState) -> int:
    if not state.dag or not state.dag.nodes:
        return 0
    depth_by_node: dict[str, int] = {}
    for node in state.dag.nodes:
        if not node.parents:
            depth_by_node[node.node_id] = 1
        else:
            depth_by_node[node.node_id] = 1 + max(depth_by_node[parent] for parent in node.parents)
    return max(depth_by_node.values(), default=0)


def _flatten_feature_outputs(state: WorkflowState) -> List[np.ndarray]:
    if not state.dag:
        return []
    outputs: List[np.ndarray] = []
    for node in state.dag.nodes:
        if node.kind not in {"feature", "multi"}:
            continue
        value = state.execution_results.get(node.node_id)
        if value is None:
            continue
        outputs.append(np.asarray(value, dtype=float).reshape(-1))
    return outputs


def _health_stats(flattened_outputs: List[np.ndarray]) -> tuple[float, float, List[str]]:
    issues: List[str] = []
    if not flattened_outputs:
        return 0.0, 0.0, ["No feature or multi outputs were available for quality evaluation."]

    concatenated = np.concatenate(flattened_outputs, axis=0) if flattened_outputs else np.asarray([], dtype=float)
    if concatenated.size == 0:
        return 0.0, 0.0, ["Feature outputs were empty after flattening."]

    nan_ratio = float(np.isnan(concatenated).sum() / concatenated.size)
    zero_variance_count = 0
    variance_eligible = 0
    for output in flattened_outputs:
        finite = output[np.isfinite(output)]
        if finite.size <= 1:
            continue
        variance_eligible += 1
        if float(np.var(finite)) <= 1e-12:
            zero_variance_count += 1
    zero_variance_ratio = float(zero_variance_count / variance_eligible) if variance_eligible else 0.0

    if nan_ratio > 0.0:
        issues.append(f"Feature outputs contain NaN values (ratio={nan_ratio:.3f}).")
    if zero_variance_ratio > 0.5:
        issues.append(f"More than half of feature outputs are near-constant (ratio={zero_variance_ratio:.3f}).")
    return nan_ratio, zero_variance_ratio, issues


def _resolve_proxy_probe_enabled(protocol: DatasetProtocol, runtime_config: Dict[str, Any]) -> bool:
    dag_quality_cfg = dict(runtime_config.get("evaluation", {}).get("dag_quality", {}))
    configured = dag_quality_cfg.get("use_proxy_probe")
    if configured is None:
        return protocol.source_mode == "real"
    return bool(configured)


def _proxy_probe_score(
    state: WorkflowState,
    protocol: DatasetProtocol,
    runtime_config: Dict[str, Any],
    catalog: OperatorCatalog,
) -> tuple[Optional[float], List[str]]:
    issues: List[str] = []
    if not state.dag or not any(node.kind == "feature" for node in state.dag.nodes):
        issues.append("Proxy probe skipped because the DAG does not contain feature nodes.")
        return None, issues

    compiled = compile_dag_for_path(
        state.dag,
        "ml",
        output_policy=str(runtime_config.get("model", {}).get("ml", {}).get("output_policy", "terminal_only")),
    )
    subset_size = int(runtime_config.get("evaluation", {}).get("dag_quality", {}).get("proxy_subset_per_split", 8))
    split_records = materialize_proxy_split_signals(protocol, subset_size)
    dataset_views = build_dataset_views(compiled, split_records, catalog)

    train_view = dataset_views["train"]
    val_view = dataset_views["val"]
    if train_view.X.size == 0 or len(set(train_view.y.tolist())) < 2:
        issues.append("Proxy probe skipped because the train subset lacks enough class diversity.")
        return None, issues
    if val_view.X.size == 0 or len(set(val_view.y.tolist())) < 2:
        issues.append("Proxy probe skipped because the validation subset lacks enough class diversity.")
        return None, issues

    baseline = run_shallow_ml_baseline(
        dataset_views,
        "logistic_regression",
        max_iter=100,
        random_state=0,
    )
    return float(baseline["metrics"]["val"]["macro_f1"]), issues


def build_dag_quality_summary(
    state: WorkflowState,
    protocol: DatasetProtocol,
    runtime_config: Dict[str, Any],
    catalog: OperatorCatalog,
) -> DagQualitySummary:
    """Build a compact quality summary for the current DAG round."""

    current_depth = _dag_depth(state)
    min_depth = int(state.data_context.get("min_depth", 2))
    max_depth = int(state.data_context.get("max_depth", 8))
    depth_ok = min_depth <= current_depth <= max_depth

    feature_node_count = 0
    multi_node_count = 0
    operator_categories: List[str] = []
    if state.dag:
        feature_node_count = sum(1 for node in state.dag.nodes if node.kind == "feature")
        multi_node_count = sum(1 for node in state.dag.nodes if node.kind == "multi")
        operator_categories = sorted({node.operator_category for node in state.dag.nodes if node.kind != "input"})

    flattened_outputs = _flatten_feature_outputs(state)
    nan_ratio, zero_variance_ratio, issues = _health_stats(flattened_outputs)
    execution_gap_count = len(state.execution_gaps)
    if execution_gap_count:
        issues.append(f"Execution gaps detected: {execution_gap_count}.")
    if not depth_ok:
        issues.append(
            f"Current depth {current_depth} is outside the target range defined by min_depth={min_depth} and max_depth={max_depth}."
        )

    proxy_probe_enabled = _resolve_proxy_probe_enabled(protocol, runtime_config)
    proxy_probe_macro_f1: Optional[float] = None
    if proxy_probe_enabled:
        proxy_probe_macro_f1, probe_issues = _proxy_probe_score(state, protocol, runtime_config, catalog)
        issues.extend(probe_issues)
        if proxy_probe_macro_f1 is not None and proxy_probe_macro_f1 < 0.55:
            issues.append(f"Proxy probe macro_f1 is weak ({proxy_probe_macro_f1:.3f}).")

    if not state.dag or not state.dag.nodes:
        recommendation_hint: RecommendationHint = "halt_candidate"
    elif execution_gap_count > 0 or current_depth > max_depth:
        recommendation_hint = "replan_candidate"
    elif (not depth_ok) or nan_ratio > 0.0:
        recommendation_hint = "patch_candidate"
    elif proxy_probe_enabled and proxy_probe_macro_f1 is not None and proxy_probe_macro_f1 < 0.55:
        recommendation_hint = "patch_candidate"
    else:
        recommendation_hint = "finish_candidate"

    return DagQualitySummary(
        current_depth=current_depth,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_ok=depth_ok,
        feature_node_count=feature_node_count,
        multi_node_count=multi_node_count,
        operator_categories=operator_categories,
        execution_gap_count=execution_gap_count,
        nan_ratio=nan_ratio,
        zero_variance_ratio=zero_variance_ratio,
        proxy_probe_enabled=proxy_probe_enabled,
        proxy_probe_macro_f1=proxy_probe_macro_f1,
        issues=issues,
        recommendation_hint=recommendation_hint,
    )
