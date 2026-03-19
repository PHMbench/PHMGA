"""Compact DAG quality evaluation for the paper-oriented workflow.

This module intentionally stays small. It does not try to become a second
training system or a platform-wide scoreboard. Its job is to summarize the
current round into a few signals that help `reflect_agent` decide whether the
DAG should finish, patch, or replan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from src.bridge import compile_dag_for_path
from src.data import DatasetProtocol, build_dataset_views, materialize_proxy_split_signals
from src.data.dataset_preparer import DatasetExecutionOptions, DatasetView
from src.model import run_shallow_ml_baseline
from src.operators import OperatorCatalog
from src.states import WorkflowState


RecommendationHint = Literal["finish_candidate", "patch_candidate", "replan_candidate", "halt_candidate"]


@dataclass
class DatasetEvidenceArtifacts:
    """Internal bundle used to avoid rematerializing dataset evidence views."""

    evidence_summary: Dict[str, Any]
    compiled_evidence_plan: Any | None
    dataset_views: Optional[Dict[str, DatasetView]]
    runtime_trace: Optional[Dict[str, Any]]


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
    dataset_level: Dict[str, Any] = Field(default_factory=dict)

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


def _resolve_dataset_evidence_path(graph_path: str) -> str:
    return "ml" if graph_path == "dag_only" else graph_path


def _resolve_dataset_evidence_output_policy(runtime_config: Dict[str, Any], graph_path: str) -> str:
    if graph_path == "torch":
        return str(runtime_config.get("model", {}).get("torch", {}).get("output_policy", "terminal_only"))
    return str(runtime_config.get("model", {}).get("ml", {}).get("output_policy", "terminal_only"))


def _proxy_probe_score_from_dataset_views(dataset_views: Dict[str, Any]) -> tuple[Optional[float], List[str]]:
    issues: List[str] = []
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


def _centroid_min_distance(dataset_views: Dict[str, Any]) -> tuple[Optional[float], List[str]]:
    issues: List[str] = []
    train_view = dataset_views["train"]
    if train_view.X.size == 0:
        issues.append("Dataset-level evidence could not assess class separation because train features are empty.")
        return None, issues
    classes = sorted(set(train_view.y.tolist()))
    if len(classes) < 2:
        issues.append("Dataset-level evidence could not assess class separation because the train split lacks class diversity.")
        return None, issues
    centroids = []
    for class_id in classes:
        class_mask = train_view.y == class_id
        class_features = train_view.X[class_mask]
        if class_features.size == 0:
            continue
        centroids.append(np.mean(class_features, axis=0))
    if len(centroids) < 2:
        issues.append("Dataset-level evidence could not build at least two class centroids.")
        return None, issues
    min_distance: Optional[float] = None
    for left_index, left_centroid in enumerate(centroids):
        for right_centroid in centroids[left_index + 1 :]:
            distance = float(np.linalg.norm(left_centroid - right_centroid))
            min_distance = distance if min_distance is None else min(min_distance, distance)
    if min_distance is None:
        issues.append("Dataset-level evidence could not compute class-centroid separation.")
        return None, issues
    if min_distance <= 1e-9:
        issues.append("Dataset-level evidence indicates near-zero class-centroid separation in the train split.")
    return min_distance, issues


def _runtime_trace_summary(runtime_trace: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not runtime_trace:
        return {
            "enabled": False,
            "split_count": 0,
            "node_count": 0,
            "approximated_node_count": 0,
            "total_elapsed_ms": 0.0,
            "slowest_node": None,
        }
    node_count = 0
    approximated_node_count = 0
    slowest_node: Optional[Dict[str, Any]] = None
    for split_payload in runtime_trace.get("splits", []):
        for node_entry in split_payload.get("nodes", []):
            node_count += 1
            if bool(node_entry.get("approximation", {}).get("enabled", False)):
                approximated_node_count += 1
            if slowest_node is None or float(node_entry.get("elapsed_ms", 0.0)) > float(slowest_node.get("elapsed_ms", 0.0)):
                slowest_node = {
                    "split": split_payload.get("split"),
                    "node_id": node_entry.get("node_id"),
                    "op_uid": node_entry.get("op_uid"),
                    "elapsed_ms": float(node_entry.get("elapsed_ms", 0.0)),
                }
    return {
        "enabled": True,
        "split_count": len(runtime_trace.get("splits", [])),
        "node_count": node_count,
        "approximated_node_count": approximated_node_count,
        "total_elapsed_ms": float(runtime_trace.get("total_elapsed_ms", 0.0)),
        "slowest_node": slowest_node,
    }


def _execute_validated_dag_np(
    state: WorkflowState,
    window: np.ndarray,
    catalog: OperatorCatalog,
) -> Dict[str, Any]:
    if state.dag is None:
        return {}
    values_by_node: Dict[str, Any] = {}
    for node in state.dag.nodes:
        if node.kind == "input":
            channel_index = int(node.params.get("channel_index", 0))
            values_by_node[node.node_id] = np.asarray(window[[channel_index], :], dtype=float)
            continue
        operator = catalog.get(node.op_uid)
        if node.kind == "multi":
            if node.input_bindings:
                binding_items = sorted(
                    node.input_bindings.items(),
                    key=lambda item: int(item[0].removeprefix("arg")),
                )
                parent_values = [values_by_node[parent_id] for _, parent_id in binding_items]
            else:
                parent_values = [values_by_node[parent_id] for parent_id in node.parents]
            values_by_node[node.node_id] = operator.forward_np(parent_values, **node.params)
            continue
        if len(node.parents) != 1:
            raise ValueError(f"Validated DAG node {node.node_id} expects one parent, got {len(node.parents)}.")
        parent_value = values_by_node[node.parents[0]]
        values_by_node[node.node_id] = operator.forward_np(parent_value, **node.params)
    return values_by_node


def _decision_summary(
    state: WorkflowState,
    split_records: Dict[str, List[Any]],
    catalog: OperatorCatalog,
) -> tuple[Dict[str, Any], List[str]]:
    issues: List[str] = []
    decision_nodes = [node for node in (state.dag.nodes if state.dag else []) if node.kind == "decision"]
    summary: Dict[str, Any] = {
        "node_count": len(decision_nodes),
        "evaluated": False,
        "split_sample_counts": {split_name: len(records) for split_name, records in split_records.items()},
        "node_summaries": {},
    }
    if not decision_nodes:
        return summary, issues

    node_summaries: Dict[str, Dict[str, Any]] = {
        node.node_id: {
            "observed_samples": 0,
            "decision_rate": None,
            "score_mean": None,
        }
        for node in decision_nodes
    }
    positive_counts: Dict[str, int] = {node.node_id: 0 for node in decision_nodes}
    score_sums: Dict[str, float] = {node.node_id: 0.0 for node in decision_nodes}
    score_counts: Dict[str, int] = {node.node_id: 0 for node in decision_nodes}

    try:
        for records in split_records.values():
            for record in records:
                values_by_node = _execute_validated_dag_np(state, record.window, catalog)
                for node in decision_nodes:
                    payload = values_by_node.get(node.node_id)
                    if not isinstance(payload, dict):
                        continue
                    node_summaries[node.node_id]["observed_samples"] += 1
                    decision_value = payload.get("decision")
                    if isinstance(decision_value, bool):
                        positive_counts[node.node_id] += int(decision_value)
                    score_value = payload.get("score")
                    if isinstance(score_value, (int, float)):
                        score_sums[node.node_id] += float(score_value)
                        score_counts[node.node_id] += 1
    except Exception as exc:
        issues.append(f"Dataset-level decision summary failed: {exc}")
        summary["evaluated"] = False
        summary["node_summaries"] = node_summaries
        return summary, issues

    for node_id, node_summary in node_summaries.items():
        observed = int(node_summary["observed_samples"])
        if observed > 0:
            node_summary["decision_rate"] = float(positive_counts[node_id] / observed)
        if score_counts[node_id] > 0:
            node_summary["score_mean"] = float(score_sums[node_id] / score_counts[node_id])
    summary["evaluated"] = True
    summary["node_summaries"] = node_summaries
    return summary, issues


def _dataset_level_evidence(
    state: WorkflowState,
    protocol: DatasetProtocol,
    runtime_config: Dict[str, Any],
    catalog: OperatorCatalog,
) -> DatasetEvidenceArtifacts:
    subset_size = int(runtime_config.get("evaluation", {}).get("dag_quality", {}).get("proxy_subset_per_split", 8))
    split_records = materialize_proxy_split_signals(protocol, subset_size)
    evidence_path = _resolve_dataset_evidence_path(state.graph_path)
    evidence: Dict[str, Any] = {
        "enabled": True,
        "source": "split_subset_execution",
        "evidence_path": evidence_path,
        "subset_size_per_split": subset_size,
        "split_window_counts": {split_name: len(records) for split_name, records in split_records.items()},
        "nonempty_splits": {},
        "feature_dims": {},
        "materialization_ok": False,
        "all_finite": False,
        "class_centroid_min_distance": None,
        "distinguishable": None,
        "decision_summary": {},
        "issues": [],
        "critical_failure": False,
    }
    runtime_trace: Optional[Dict[str, Any]] = None

    if not state.dag or not any(node.kind in {"feature", "multi"} for node in state.dag.nodes):
        evidence["issues"].append("Dataset-level evidence skipped because the DAG does not yet expose feature or multi outputs.")
        evidence["decision_summary"], decision_issues = _decision_summary(state, split_records, catalog)
        evidence["issues"].extend(decision_issues)
        evidence["runtime_trace_summary"] = _runtime_trace_summary(runtime_trace)
        return DatasetEvidenceArtifacts(
            evidence_summary=evidence,
            compiled_evidence_plan=None,
            dataset_views=None,
            runtime_trace=runtime_trace,
        )

    compiled = None
    dataset_views: Optional[Dict[str, DatasetView]] = None
    try:
        compiled = compile_dag_for_path(
            state.dag,
            evidence_path,
            output_policy=_resolve_dataset_evidence_output_policy(runtime_config, evidence_path),
        )
        runtime_trace = {}
        execution_options = DatasetExecutionOptions(
            mode="dataset_evidence",
            enable_runtime_trace=True,
            dataset_name=protocol.dataset_name,
            graph_path=state.graph_path,
            evidence_path=evidence_path,
            sample_budget=subset_size,
            cross_correlation_large_input_threshold=2048,
            cross_correlation_max_lag=256,
        )
        dataset_views = build_dataset_views(
            compiled,
            split_records,
            catalog,
            backend="np",
            execution_options=execution_options,
            runtime_trace=runtime_trace,
        )
    except Exception as exc:
        evidence["critical_failure"] = True
        evidence["issues"].append(f"Dataset-level materialization failed for path '{evidence_path}': {exc}")
        evidence["decision_summary"], decision_issues = _decision_summary(state, split_records, catalog)
        evidence["issues"].extend(decision_issues)
        evidence["runtime_trace_summary"] = _runtime_trace_summary(runtime_trace)
        if runtime_trace:
            evidence["runtime_trace_artifact"] = runtime_trace
        return DatasetEvidenceArtifacts(
            evidence_summary=evidence,
            compiled_evidence_plan=compiled,
            dataset_views=dataset_views,
            runtime_trace=runtime_trace,
        )

    nonempty_splits: Dict[str, bool] = {}
    feature_dims: Dict[str, int] = {}
    all_finite = True
    for split_name, view in dataset_views.items():
        feature_dims[split_name] = int(view.X.shape[1]) if view.X.ndim == 2 else 0
        nonempty_splits[split_name] = bool(view.X.ndim == 2 and view.X.shape[0] > 0 and view.X.shape[1] > 0)
        if view.X.size and not np.isfinite(view.X).all():
            all_finite = False
    evidence["nonempty_splits"] = nonempty_splits
    evidence["feature_dims"] = feature_dims
    evidence["all_finite"] = all_finite

    if not all(nonempty_splits.values()):
        evidence["issues"].append("Dataset-level evidence found empty feature materialization in at least one split.")
    if not all_finite:
        evidence["issues"].append("Dataset-level evidence found non-finite values in materialized split features.")

    nonzero_dims = {dimension for dimension in feature_dims.values() if dimension > 0}
    if len(nonzero_dims) > 1:
        evidence["issues"].append("Dataset-level evidence found inconsistent feature dimensions across splits.")

    centroid_min_distance, centroid_issues = _centroid_min_distance(dataset_views)
    evidence["class_centroid_min_distance"] = centroid_min_distance
    evidence["issues"].extend(centroid_issues)
    if centroid_min_distance is not None:
        evidence["distinguishable"] = bool(centroid_min_distance > 1e-9)

    evidence["decision_summary"], decision_issues = _decision_summary(state, split_records, catalog)
    evidence["issues"].extend(decision_issues)
    evidence["materialization_ok"] = bool(all(nonempty_splits.values()) and all_finite and nonzero_dims)
    evidence["runtime_trace_summary"] = _runtime_trace_summary(runtime_trace)
    if runtime_trace:
        evidence["runtime_trace_artifact"] = runtime_trace
    return DatasetEvidenceArtifacts(
        evidence_summary=evidence,
        compiled_evidence_plan=compiled,
        dataset_views=dataset_views,
        runtime_trace=runtime_trace,
    )


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

    dataset_evidence = _dataset_level_evidence(state, protocol, runtime_config, catalog)
    dataset_level = dataset_evidence.evidence_summary
    issues.extend(list(dataset_level.get("issues", [])))

    proxy_probe_enabled = _resolve_proxy_probe_enabled(protocol, runtime_config)
    proxy_probe_macro_f1: Optional[float] = None
    if (
        proxy_probe_enabled
        and not dataset_level.get("critical_failure", False)
        and dataset_evidence.dataset_views is not None
    ):
        proxy_probe_macro_f1, probe_issues = _proxy_probe_score_from_dataset_views(dataset_evidence.dataset_views)
        issues.extend(probe_issues)
        if proxy_probe_macro_f1 is not None and proxy_probe_macro_f1 < 0.55:
            issues.append(f"Proxy probe macro_f1 is weak ({proxy_probe_macro_f1:.3f}).")
    dataset_level["proxy_probe_enabled"] = proxy_probe_enabled
    dataset_level["proxy_probe_macro_f1"] = proxy_probe_macro_f1

    if not state.dag or not state.dag.nodes:
        recommendation_hint: RecommendationHint = "halt_candidate"
    elif execution_gap_count > 0 or current_depth > max_depth or bool(dataset_level.get("critical_failure", False)):
        recommendation_hint = "replan_candidate"
    elif (not depth_ok) or nan_ratio > 0.0 or not bool(dataset_level.get("materialization_ok", True)):
        recommendation_hint = "patch_candidate"
    elif dataset_level.get("distinguishable") is False:
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
        dataset_level=dataset_level,
        issues=issues,
        recommendation_hint=recommendation_hint,
    )
