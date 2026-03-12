"""DAG intermediate representation and validation utilities."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import networkx as nx
from pydantic import BaseModel, Field, model_validator


BackendName = Literal["np", "pt", "sym"]
NodeKind = Literal["input", "transform", "feature", "artifact"]
ExecutionRole = Literal["trainable", "fixed", "proxy", "outer_only"]


class DagNode(BaseModel):
    """Canonical DAG node passed from workflow front-end into the bridge."""
    node_id: str
    op_uid: str
    name: str
    kind: NodeKind
    params: Dict[str, Any] = Field(default_factory=dict)
    parents: List[str] = Field(default_factory=list)
    in_shape: List[int]
    out_shape: List[int]
    backend_availability: List[BackendName]
    execution_role: ExecutionRole

    @model_validator(mode="after")
    def ensure_shapes(self) -> "DagNode":
        """Require explicit shape contracts on every node."""
        if not self.in_shape or not self.out_shape:
            raise ValueError("Shapes must be non-empty")
        return self


class DagEdge(BaseModel):
    """Explicit edge record for JSON export and manifest generation."""
    source: str
    target: str


class DagJson(BaseModel):
    """Validated node-link style DAG payload."""
    nodes: List[DagNode]
    edges: List[DagEdge]


def validate_dag_json(payload: Dict[str, Any] | DagJson) -> DagJson:
    """Validate DAG shape, references, and acyclicity before bridge entry."""
    dag = payload if isinstance(payload, DagJson) else DagJson.model_validate(payload)
    graph = nx.DiGraph()
    node_ids = {node.node_id for node in dag.nodes}
    if len(node_ids) != len(dag.nodes):
        raise ValueError("Duplicate node ids are not allowed")
    for node in dag.nodes:
        graph.add_node(node.node_id)
        for parent in node.parents:
            if parent not in node_ids:
                raise ValueError(f"Parent {parent} missing from DAG")
            graph.add_edge(parent, node.node_id)
    for edge in dag.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            raise ValueError("Edges must reference existing nodes")
        graph.add_edge(edge.source, edge.target)
    # The rebuilt repo treats a validated DAG JSON as the sole legal hand-off
    # between workflow and backend compilation, so cycles are rejected here.
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("DAG must be acyclic")
    return dag


class DAGTracker:
    """Incremental DAG builder used by the workflow layer."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._nodes: Dict[str, DagNode] = {}

    def add_node(self, node: DagNode) -> None:
        """Insert one node and reject it immediately if it introduces a cycle."""
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id)
        for parent in node.parents:
            self._graph.add_edge(parent, node.node_id)
        if not nx.is_directed_acyclic_graph(self._graph):
            self._graph.remove_node(node.node_id)
            self._nodes.pop(node.node_id, None)
            raise ValueError(f"Adding {node.node_id} would create a cycle")

    def export(self) -> DagJson:
        """Export a topologically ordered DAG JSON and re-validate it."""
        edges = [DagEdge(source=source, target=target) for source, target in self._graph.edges()]
        ordered_nodes = [self._nodes[node_id] for node_id in nx.topological_sort(self._graph)]
        return validate_dag_json({"nodes": [node.model_dump() for node in ordered_nodes], "edges": [edge.model_dump() for edge in edges]})
