from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import networkx as nx
import numpy as np

from src.data.protocol import DatasetProtocol, SignalRecord, materialize_split_signals
from src.states.phm_states import InputData, PHMState, ProcessedData
from src.tools import MultiVariableOp, get_operator


SplitMap = Dict[str, Dict[str, Any]]


@dataclass
class ReplayResult:
    state: PHMState
    split_keys: tuple[str, ...]
    labels_by_split: Dict[str, Dict[str, int]]


def _normalize_parent_ids(parents: List[str] | str) -> List[str]:
    if isinstance(parents, list):
        return [str(parent) for parent in parents if str(parent).strip()]
    if str(parents).strip():
        return [segment.strip() for segment in str(parents).split(",") if segment.strip()]
    return []


def _signal_to_channel_window(window: np.ndarray, channel_index: int) -> np.ndarray:
    value = np.asarray(window[channel_index, :], dtype=float)
    return value.reshape(1, -1, 1)


def build_root_split_maps(protocol: DatasetProtocol) -> tuple[Dict[str, SplitMap], Dict[str, Dict[str, int]]]:
    """Materialize split-before-windowing signals into per-channel DAG inputs."""
    split_records = materialize_split_signals(protocol)
    channel_count = int(protocol.samples[0].channels)
    channel_names = [f"ch{index + 1}" for index in range(channel_count)]

    roots: Dict[str, SplitMap] = {
        channel_name: {split_name: {} for split_name in split_records}
        for channel_name in channel_names
    }
    labels_by_split: Dict[str, Dict[str, int]] = {split_name: {} for split_name in split_records}

    for split_name, records in split_records.items():
        for record in records:
            labels_by_split[split_name][record.window_id] = int(record.label)
            for channel_index, channel_name in enumerate(channel_names):
                roots[channel_name][split_name][record.window_id] = _signal_to_channel_window(
                    record.window,
                    channel_index,
                )
    return roots, labels_by_split


def build_input_split_results(
    split_records: Dict[str, List[SignalRecord]],
    *,
    channel_ids: List[str] | None = None,
) -> Dict[str, SplitMap]:
    """Build per-channel split maps directly from split records."""
    first_record = next(iter(next(iter(split_records.values()))), None)
    if first_record is None:
        raise ValueError("split_records must contain at least one SignalRecord.")
    channel_count = int(first_record.window.shape[0])
    resolved_channel_ids = channel_ids or [f"ch{index + 1}" for index in range(channel_count)]
    if len(resolved_channel_ids) != channel_count:
        raise ValueError(
            f"channel_ids count {len(resolved_channel_ids)} does not match split record channel count {channel_count}."
        )

    outputs: Dict[str, SplitMap] = {
        channel_id: {split_name: {} for split_name in split_records}
        for channel_id in resolved_channel_ids
    }
    for split_name, records in split_records.items():
        for record in records:
            for channel_index, channel_id in enumerate(resolved_channel_ids):
                outputs[channel_id][split_name][record.window_id] = _signal_to_channel_window(
                    record.window,
                    channel_index,
                )
    return outputs


def build_split_labels(split_records: Dict[str, List[SignalRecord]]) -> Dict[str, Dict[str, int]]:
    return {
        split_name: {
            record.window_id: int(record.label)
            for record in records
        }
        for split_name, records in split_records.items()
    }


def _node_results_by_split(node: InputData | ProcessedData, split_names: Iterable[str]) -> SplitMap:
    raw = node.results if isinstance(node.results, dict) else {}
    return {
        split_name: dict(raw.get(split_name) or {})
        for split_name in split_names
    }


def execute_single_parent_operator(
    op: Any,
    *,
    parent_id: str,
    nodes: Dict[str, InputData | ProcessedData],
    split_names: Iterable[str],
) -> SplitMap:
    outputs: SplitMap = {}
    parent_results = _node_results_by_split(nodes[parent_id], split_names)
    for split_name in split_names:
        split_values = parent_results.get(split_name, {})
        outputs[split_name] = {
            sample_id: op.execute(value)
            for sample_id, value in split_values.items()
            if value is not None
        }
    return outputs


def execute_multi_parent_operator(
    op: Any,
    *,
    parent_ids: List[str],
    nodes: Dict[str, InputData | ProcessedData],
    split_names: Iterable[str],
) -> SplitMap:
    outputs: SplitMap = {}
    for split_name in split_names:
        parent_maps = {
            parent_id: _node_results_by_split(nodes[parent_id], split_names).get(split_name, {})
            for parent_id in parent_ids
        }
        if not parent_maps:
            outputs[split_name] = {}
            continue
        common_ids = set.intersection(*(set(values.keys()) for values in parent_maps.values()))
        split_outputs: Dict[str, Any] = {}
        for sample_id in sorted(common_ids):
            payload = {parent_id: parent_maps[parent_id][sample_id] for parent_id in parent_ids}
            split_outputs[sample_id] = op.execute(payload)
        outputs[split_name] = split_outputs
    return outputs


def replay_state_on_split_maps(
    state: PHMState,
    *,
    root_split_maps: Dict[str, SplitMap],
    labels_by_split: Dict[str, Dict[str, int]],
    split_names: Iterable[str] = ("train", "val", "test"),
) -> PHMState:
    """Replay an already-built DAG on train/val/test split maps."""
    working_state = state.model_copy(deep=True)
    split_names = tuple(split_names)
    new_leaves: list[str] = []

    for channel_name in working_state.dag_state.channels:
        node = working_state.dag_state.nodes[channel_name]
        if not isinstance(node, InputData):
            raise TypeError(f"Expected input node for channel {channel_name}, got {type(node)!r}")
        if channel_name not in root_split_maps:
            raise KeyError(f"Missing split map for root channel {channel_name}")
        all_labels = {
            sample_id: label
            for split_label_map in labels_by_split.values()
            for sample_id, label in split_label_map.items()
        }
        node.results = {split_name: dict(root_split_maps[channel_name].get(split_name, {})) for split_name in split_names}
        node.meta = dict(node.meta)
        node.meta["labels"] = all_labels
        node.meta["labels_by_split"] = {key: dict(value) for key, value in labels_by_split.items()}

    for node_id, node in working_state.dag_state.nodes.items():
        if isinstance(node, ProcessedData):
            node.results = {}

    tracker = working_state.tracker()
    for node_id in nx.topological_sort(tracker.g):
        node = working_state.dag_state.nodes[node_id]
        if isinstance(node, InputData):
            if node_id not in new_leaves:
                new_leaves.append(node_id)
            continue

        parent_ids = _normalize_parent_ids(node.parents)
        if not parent_ids:
            working_state.dag_state.error_log.append(f"Leaf replay skipped for {node_id}: empty parents.")
            continue

        tool_name = str(node.meta.get("tool") or node.meta.get("method") or node.method or "").strip()
        if not tool_name:
            working_state.dag_state.error_log.append(f"Leaf replay skipped for {node_id}: missing tool.")
            continue

        try:
            op_cls = get_operator(tool_name)
            params = dict(node.meta.get("params") or {})
            if "fs" in getattr(op_cls, "model_fields", {}) and "fs" not in params and working_state.fs is not None:
                params["fs"] = working_state.fs
            op = op_cls(**params, parent=node.meta.get("parent"))
            if issubclass(op_cls, MultiVariableOp):
                outputs = execute_multi_parent_operator(
                    op,
                    parent_ids=parent_ids,
                    nodes=working_state.dag_state.nodes,
                    split_names=split_names,
                )
            else:
                outputs = execute_single_parent_operator(
                    op,
                    parent_id=parent_ids[0],
                    nodes=working_state.dag_state.nodes,
                    split_names=split_names,
                )
            node.results = outputs
            for parent_id in parent_ids:
                if parent_id in new_leaves:
                    new_leaves.remove(parent_id)
            new_leaves.append(node_id)
        except Exception as exc:  # pragma: no cover - exercised by real runs
            working_state.dag_state.error_log.append(f"Replay failed for node {node_id}: {exc}")

    working_state.dag_state.leaves = new_leaves or list(working_state.dag_state.leaves)
    working_state._tracker_instance = None
    return working_state


def replay_state_on_split_results(
    state: PHMState,
    split_results: Dict[str, SplitMap],
    *,
    labels_by_split: Dict[str, Dict[str, int]] | None = None,
    split_keys: Iterable[str] = ("train", "val", "test"),
) -> ReplayResult:
    """Compatibility wrapper returning both replayed state and split metadata."""
    resolved_labels = labels_by_split or {split_name: {} for split_name in split_keys}
    replayed_state = replay_state_on_split_maps(
        state,
        root_split_maps=split_results,
        labels_by_split=resolved_labels,
        split_names=split_keys,
    )
    return ReplayResult(
        state=replayed_state,
        split_keys=tuple(split_keys),
        labels_by_split=resolved_labels,
    )


def summarize_branch(state: PHMState, leaf_id: str) -> str:
    """Return a compact operator-path summary for a terminal leaf."""
    graph = state.tracker().g
    if leaf_id not in graph:
        return leaf_id

    ancestors = nx.ancestors(graph, leaf_id)
    ordered = [node_id for node_id in nx.topological_sort(graph) if node_id in ancestors or node_id == leaf_id]
    methods: list[str] = []
    for node_id in ordered:
        node = state.dag_state.nodes[node_id]
        method = str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", "") or "").strip()
        if method:
            methods.append(method)
    deduped = list(dict.fromkeys(methods))
    return " -> ".join(deduped) if deduped else leaf_id


def _ordered_sample_ids(feature_map: Dict[str, Any], label_map: Dict[str, int]) -> list[str]:
    if not feature_map or not label_map:
        return []
    return [sample_id for sample_id in sorted(feature_map) if sample_id in label_map]


def _flatten_features(feature_map: Dict[str, Any], label_map: Dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    if not feature_map or not label_map:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    sample_ids = _ordered_sample_ids(feature_map, label_map)
    if not sample_ids:
        return np.empty((0, 0)), np.empty((0,), dtype=int)

    features = []
    labels = []
    for sample_id in sample_ids:
        array = np.asarray(feature_map[sample_id], dtype=float).reshape(-1)
        features.append(array)
        labels.append(int(label_map[sample_id]))
    return np.vstack(features), np.asarray(labels, dtype=int)


def extract_leaf_datasets(
    state: PHMState,
    *,
    labels_by_split: Dict[str, Dict[str, int]],
    split_names: Iterable[str] = ("train", "val", "test"),
) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Build shallow-ML-ready datasets for each terminal leaf."""
    datasets: Dict[str, Dict[str, Any]] = {}
    failures: list[Dict[str, Any]] = []

    for leaf_id in state.dag_state.leaves:
        node = state.dag_state.nodes.get(leaf_id)
        if node is None:
            continue
        if isinstance(node, InputData):
            failures.append(
                {
                    "leaf_id": leaf_id,
                    "branch_id": leaf_id,
                    "branch_summary": leaf_id,
                    "failure_reason": "input leaf is not a processed feature branch",
                }
            )
            continue

        split_results = _node_results_by_split(node, split_names)
        train_window_ids = _ordered_sample_ids(split_results.get("train", {}), labels_by_split.get("train", {}))
        val_window_ids = _ordered_sample_ids(split_results.get("val", {}), labels_by_split.get("val", {}))
        test_window_ids = _ordered_sample_ids(split_results.get("test", {}), labels_by_split.get("test", {}))
        train_X, train_y = _flatten_features(split_results.get("train", {}), labels_by_split.get("train", {}))
        val_X, val_y = _flatten_features(split_results.get("val", {}), labels_by_split.get("val", {}))
        test_X, test_y = _flatten_features(split_results.get("test", {}), labels_by_split.get("test", {}))

        feature_dim = int(train_X.shape[1]) if train_X.size else 0
        failure_reason = ""
        if train_X.size == 0 or test_X.size == 0:
            failure_reason = "missing train/test features after DAG replay"
        elif feature_dim == 0:
            failure_reason = "empty feature dimensionality"
        elif val_X.size and int(val_X.shape[1]) != feature_dim:
            failure_reason = "validation feature dimensionality mismatch"
        elif int(test_X.shape[1]) != feature_dim:
            failure_reason = "test feature dimensionality mismatch"

        if failure_reason:
            failures.append(
                {
                    "leaf_id": leaf_id,
                    "branch_id": leaf_id,
                    "branch_summary": summarize_branch(state, leaf_id),
                    "failure_reason": failure_reason,
                }
            )
            continue

        datasets[leaf_id] = {
            "X_train": train_X,
            "y_train": train_y,
            "X_val": val_X,
            "y_val": val_y,
            "X_test": test_X,
            "y_test": test_y,
            "branch_id": leaf_id,
            "leaf_id": leaf_id,
            "branch_summary": summarize_branch(state, leaf_id),
            "feature_dim": feature_dim,
            "train_samples": int(train_X.shape[0]),
            "val_samples": int(val_X.shape[0]),
            "test_samples": int(test_X.shape[0]),
            "train_window_ids": train_window_ids,
            "val_window_ids": val_window_ids,
            "test_window_ids": test_window_ids,
        }

    return datasets, failures
