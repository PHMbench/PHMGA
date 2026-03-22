"""Feature-plan extraction and split materialization from the built DAG."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field

from src.data import SignalRecord
from src.states.phm_states import DAGState, InputData, ProcessedData
from src.tools import AggregateOp, MultiVariableOp, get_operator


def _preview_array(node: InputData | ProcessedData) -> Optional[np.ndarray]:
    results = getattr(node, "results", {}) or {}
    for split_name in ("ref", "tst"):
        payload = results.get(split_name)
        if isinstance(payload, dict) and payload:
            first = next(iter(payload.values()))
            return np.asarray(first)
        if isinstance(payload, np.ndarray):
            return np.asarray(payload)
    return None


class FeatureBranch(BaseModel):
    node_id: str
    op_name: str
    parents: List[str]
    dimension: int
    preview_shape: List[int]


class FeaturePlan(BaseModel):
    branches: List[FeatureBranch] = Field(default_factory=list)
    topo_order: List[str] = Field(default_factory=list)
    needed_node_ids: List[str] = Field(default_factory=list)


def build_feature_plan(dag_state: DAGState) -> FeaturePlan:
    graph = nx.DiGraph()
    for node_id, node in dag_state.nodes.items():
        graph.add_node(node_id)
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        for parent in parents:
            if parent:
                graph.add_edge(parent, node_id)

    memo: Dict[str, bool] = {}

    def is_feature_node(node_id: str) -> bool:
        if node_id in memo:
            return memo[node_id]
        node = dag_state.nodes[node_id]
        if isinstance(node, InputData):
            memo[node_id] = False
            return False
        op_name = str(node.meta.get("tool") or node.meta.get("method") or node.method)
        op_cls = get_operator(op_name)
        if issubclass(op_cls, AggregateOp):
            memo[node_id] = True
            return True
        if issubclass(op_cls, MultiVariableOp):
            parents = node.parents if isinstance(node.parents, list) else [node.parents]
            memo[node_id] = all(is_feature_node(parent) for parent in parents)
            return memo[node_id]
        memo[node_id] = False
        return False

    branches: List[FeatureBranch] = []
    for leaf_id in dag_state.leaves:
        if not is_feature_node(leaf_id):
            continue
        node = dag_state.nodes[leaf_id]
        preview = _preview_array(node)
        if preview is None:
            continue
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        branches.append(
            FeatureBranch(
                node_id=leaf_id,
                op_name=str(node.meta.get("tool") or node.meta.get("method") or node.method),
                parents=list(parents),
                dimension=int(np.asarray(preview).reshape(-1).shape[0]),
                preview_shape=[int(dim) for dim in np.asarray(preview).shape],
            )
        )

    if not branches:
        raise ValueError("No terminal feature branches were found in the current DAG.")

    needed: set[str] = set()
    queue = deque(branch.node_id for branch in branches)
    while queue:
        node_id = queue.popleft()
        if node_id in needed:
            continue
        needed.add(node_id)
        node = dag_state.nodes[node_id]
        parents = node.parents if isinstance(node.parents, list) else [node.parents]
        queue.extend(parent for parent in parents if parent)

    topo_order = [node_id for node_id in nx.topological_sort(graph) if node_id in needed]
    return FeaturePlan(
        branches=branches,
        topo_order=topo_order,
        needed_node_ids=topo_order,
    )


def _execute_single_node(
    node_id: str,
    dag_state: DAGState,
    channel_index: Dict[str, int],
    values_by_node: Dict[str, np.ndarray],
    window: np.ndarray,
) -> np.ndarray:
    node = dag_state.nodes[node_id]
    if isinstance(node, InputData):
        index = channel_index[node_id]
        return window[[index], :].T.reshape(1, window.shape[1], 1)

    op_name = str(node.meta.get("tool") or node.meta.get("method") or node.method)
    op_cls = get_operator(op_name)
    params = dict(node.meta.get("params", {}))
    parent_ids = node.parents if isinstance(node.parents, list) else [node.parents]
    op = op_cls(**params, parent=parent_ids)
    if issubclass(op_cls, MultiVariableOp):
        payload = {parent_id: values_by_node[parent_id] for parent_id in parent_ids}
        return np.asarray(op.execute(payload))
    if len(parent_ids) != 1:
        raise ValueError(f"Single-input op {op_name} expected one parent, got {parent_ids}.")
    return np.asarray(op.execute(values_by_node[parent_ids[0]]))


def materialize_feature_views(
    dag_state: DAGState,
    feature_plan: FeaturePlan,
    split_records: Dict[str, List[SignalRecord]],
) -> Dict[str, Dict[str, Any]]:
    channel_index = {channel_name: index for index, channel_name in enumerate(dag_state.channels)}
    outputs: Dict[str, Dict[str, Any]] = {}

    for split_name, records in split_records.items():
        branch_features = {branch.node_id: [] for branch in feature_plan.branches}
        labels: List[int] = []
        sample_ids: List[str] = []
        for record in records:
            values_by_node: Dict[str, np.ndarray] = {}
            for node_id in feature_plan.topo_order:
                values_by_node[node_id] = _execute_single_node(
                    node_id,
                    dag_state,
                    channel_index,
                    values_by_node,
                    np.asarray(record.window, dtype=float),
                )
            for branch in feature_plan.branches:
                branch_features[branch.node_id].append(values_by_node[branch.node_id].reshape(-1))
            labels.append(int(record.label))
            sample_ids.append(record.window_id)

        outputs[split_name] = {
            "labels": np.asarray(labels, dtype=int),
            "sample_ids": sample_ids,
            "branches": {
                node_id: (
                    np.vstack(values) if values else np.zeros((0, 0), dtype=float)
                )
                for node_id, values in branch_features.items()
            },
        }
    return outputs
