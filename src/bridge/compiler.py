"""Bridge compiler from validated DAG JSON to graph-path-specific plans."""

from __future__ import annotations

from typing import Dict, List, Literal, Set, Union

from pydantic import BaseModel, Field

from src.dag import DagJson, DagNode, validate_dag_json
from src.utils import hash_payload


OutputPolicy = Literal["terminal_only", "include_intermediate_features"]


class ManifestNode(BaseModel):
    """Compact per-node summary stored in compiled manifests."""

    node_id: str
    op_uid: str
    kind: str
    operator_category: str
    rank_class: str
    legal_paths: List[str]
    backend_availability: List[str]
    shape_inference: Dict[str, List[int]]


class CompiledDagManifest(BaseModel):
    """Stable manifest shared by all graph-path outputs."""

    dag_hash: str
    topo_order: List[str]
    nodes: List[ManifestNode]
    path_type: Literal["dag_only", "ml", "torch"]
    warnings: List[str] = Field(default_factory=list)


class DagArtifacts(BaseModel):
    """Artifact bundle for the ``dag_only`` path."""

    node_inventory: List[Dict[str, str]]
    edge_inventory: List[Dict[str, str]]
    method_description: str
    manifest: CompiledDagManifest


class CompiledExecutionNode(BaseModel):
    """Path-specific execution node copied from validated DAG contracts."""

    node_id: str
    op_uid: str
    kind: Literal["input", "transform", "feature", "multi"]
    parents: List[str]
    input_bindings: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, object] = Field(default_factory=dict)
    channel_index: int | None = None


class CompiledOutputSpec(BaseModel):
    """One final output node selected by bridge output policy."""

    output_node_id: str
    output_kind: Literal["feature", "multi"]


class FeaturePipelinePlan(BaseModel):
    """Bridge output for the lightweight ML path."""

    execution_nodes: List[CompiledExecutionNode]
    output_specs: List[CompiledOutputSpec]
    output_policy: OutputPolicy
    manifest: CompiledDagManifest


class ModelBuildPlan(BaseModel):
    """Bridge output for the trainable path."""

    backend_target: str
    execution_nodes: List[CompiledExecutionNode]
    output_specs: List[CompiledOutputSpec]
    output_policy: OutputPolicy
    trainable_head: Dict[str, int]
    manifest: CompiledDagManifest


def _build_manifest(dag: DagJson, path_type: Literal["dag_only", "ml", "torch"]) -> CompiledDagManifest:
    """Derive a stable manifest that all downstream paths can report on."""

    validated = validate_dag_json(dag)
    topo_order = [node.node_id for node in validated.nodes]
    nodes = [
        ManifestNode(
            node_id=node.node_id,
            op_uid=node.op_uid,
            kind=node.kind,
            operator_category=node.operator_category,
            rank_class=node.rank_class,
            legal_paths=node.legal_paths,
            backend_availability=node.backend_availability,
            shape_inference={"in": node.in_shape, "out": node.out_shape},
        )
        for node in validated.nodes
    ]
    warnings = []
    if not any(node.kind == "feature" for node in validated.nodes):
        warnings.append("No feature nodes detected.")
    if path_type in {"ml", "torch"} and not any(node.op_uid.startswith("feature.") for node in validated.nodes):
        warnings.append("Downstream path requested without explicit feature operators.")
    return CompiledDagManifest(
        dag_hash=hash_payload(validated.model_dump()),
        topo_order=topo_order,
        nodes=nodes,
        path_type=path_type,
        warnings=warnings,
    )


def _candidate_output_nodes(dag: DagJson) -> List[DagNode]:
    return [
        node
        for node in dag.nodes
        if node.kind in {"feature", "multi"} and any(path in {"ml", "torch"} for path in node.legal_paths)
    ]


def _select_output_specs(dag: DagJson, output_policy: OutputPolicy) -> List[CompiledOutputSpec]:
    """Select final output nodes without changing the validated DAG itself."""

    candidates = _candidate_output_nodes(dag)
    candidate_ids = {node.node_id for node in candidates}
    children_by_parent: Dict[str, Set[str]] = {}
    for node in dag.nodes:
        for parent in node.parents:
            children_by_parent.setdefault(parent, set()).add(node.node_id)

    output_nodes: List[DagNode] = []
    for node in candidates:
        candidate_children = children_by_parent.get(node.node_id, set()) & candidate_ids
        if output_policy == "terminal_only":
            if candidate_children:
                continue
            output_nodes.append(node)
            continue
        if node.kind == "feature":
            output_nodes.append(node)
            continue
        if node.kind == "multi" and not candidate_children:
            output_nodes.append(node)

    return [
        CompiledOutputSpec(output_node_id=node.node_id, output_kind=node.kind)
        for node in output_nodes
    ]


def _ancestor_closure(node_lookup: Dict[str, DagNode], output_specs: List[CompiledOutputSpec]) -> Set[str]:
    needed: Set[str] = set()
    stack = [spec.output_node_id for spec in output_specs]
    while stack:
        node_id = stack.pop()
        if node_id in needed:
            continue
        needed.add(node_id)
        stack.extend(node_lookup[node_id].parents)
    return needed


def _build_execution_nodes(dag: DagJson, output_specs: List[CompiledOutputSpec]) -> List[CompiledExecutionNode]:
    """Build the minimal executable subgraph required for selected outputs."""

    node_lookup = {node.node_id: node for node in dag.nodes}
    needed_ids = _ancestor_closure(node_lookup, output_specs)
    execution_nodes: List[CompiledExecutionNode] = []
    for node in dag.nodes:
        if node.node_id not in needed_ids:
            continue
        if node.kind == "decision":
            continue
        if node.kind == "artifact":
            continue
        channel_index = None
        if node.kind == "input":
            channel_index = int(node.params["channel_index"])
        execution_nodes.append(
            CompiledExecutionNode(
                node_id=node.node_id,
                op_uid=node.op_uid,
                kind=node.kind,
                parents=list(node.parents),
                input_bindings=dict(node.input_bindings),
                params=dict(node.params),
                channel_index=channel_index,
            )
        )
    return execution_nodes


def _product(shape: List[int]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size


def _infer_input_dim(dag: DagJson, output_specs: List[CompiledOutputSpec]) -> int:
    node_lookup = {node.node_id: node for node in dag.nodes}
    return sum(_product(node_lookup[spec.output_node_id].out_shape) for spec in output_specs)


def compile_dag_for_path(
    dag: DagJson,
    path_type: Literal["dag_only", "ml", "torch"],
    *,
    output_policy: OutputPolicy = "terminal_only",
) -> Union[DagArtifacts, FeaturePipelinePlan, ModelBuildPlan]:
    """Compile one validated DAG into the selected graph-path backend object."""

    validated = validate_dag_json(dag)
    manifest = _build_manifest(validated, path_type)
    if path_type == "dag_only":
        return DagArtifacts(
            node_inventory=[{"node_id": node.node_id, "op_uid": node.op_uid, "kind": node.kind} for node in validated.nodes],
            edge_inventory=[{"source": edge.source, "target": edge.target} for edge in validated.edges],
            method_description="DAG structural prior generated from the canonical operator catalog.",
            manifest=manifest,
        )

    output_specs = _select_output_specs(validated, output_policy)
    execution_nodes = _build_execution_nodes(validated, output_specs)
    if path_type == "ml":
        return FeaturePipelinePlan(
            execution_nodes=execution_nodes,
            output_specs=output_specs,
            output_policy=output_policy,
            manifest=manifest,
        )

    input_dim = _infer_input_dim(validated, output_specs)
    return ModelBuildPlan(
        backend_target="torch",
        execution_nodes=execution_nodes,
        output_specs=output_specs,
        output_policy=output_policy,
        trainable_head={"input_dim": input_dim, "hidden_dim": 8, "output_dim": 2},
        manifest=manifest,
    )
