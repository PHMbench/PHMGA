from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, Field

from src.dag import DagJson, DagNode, validate_dag_json
from src.utils import hash_payload


class ManifestNode(BaseModel):
    node_id: str
    op_uid: str
    kind: str
    backend_availability: List[str]
    shape_inference: Dict[str, List[int]]


class CompiledDagManifest(BaseModel):
    dag_hash: str
    topo_order: List[str]
    nodes: List[ManifestNode]
    path_type: Literal["dag_only", "ml", "torch"]
    warnings: List[str] = Field(default_factory=list)


class DagArtifacts(BaseModel):
    node_inventory: List[Dict[str, str]]
    edge_inventory: List[Dict[str, str]]
    method_description: str
    manifest: CompiledDagManifest


class FeatureSpec(BaseModel):
    feature_node_id: str
    channel_index: int
    transform_ops: List[str]
    feature_op: str


class FeaturePipelinePlan(BaseModel):
    feature_specs: List[FeatureSpec]
    manifest: CompiledDagManifest


class ModelBuildPlan(BaseModel):
    backend_target: str
    feature_specs: List[FeatureSpec]
    trainable_head: Dict[str, int]
    manifest: CompiledDagManifest


def _build_manifest(dag: DagJson, path_type: Literal["dag_only", "ml", "torch"]) -> CompiledDagManifest:
    validated = validate_dag_json(dag)
    node_lookup = {node.node_id: node for node in validated.nodes}
    topo_order = [node.node_id for node in validated.nodes]
    nodes = [
        ManifestNode(
            node_id=node.node_id,
            op_uid=node.op_uid,
            kind=node.kind,
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


def _lineage(node_lookup: Dict[str, DagNode], node_id: str) -> List[DagNode]:
    node = node_lookup[node_id]
    lineage: list[DagNode] = []
    current = node
    while current.parents:
        parent = node_lookup[current.parents[0]]
        lineage.append(parent)
        current = parent
    lineage.reverse()
    return lineage


def _build_feature_specs(dag: DagJson) -> List[FeatureSpec]:
    node_lookup = {node.node_id: node for node in dag.nodes}
    feature_specs: list[FeatureSpec] = []
    for node in dag.nodes:
        if node.kind != "feature":
            continue
        lineage = _lineage(node_lookup, node.node_id)
        input_node = next(parent for parent in lineage if parent.kind == "input")
        transform_ops = [parent.op_uid for parent in lineage if parent.kind == "transform"]
        feature_specs.append(
            FeatureSpec(
                feature_node_id=node.node_id,
                channel_index=int(input_node.params["channel_index"]),
                transform_ops=transform_ops,
                feature_op=node.op_uid,
            )
        )
    return feature_specs


def compile_dag_for_path(
    dag: DagJson,
    path_type: Literal["dag_only", "ml", "torch"],
) -> DagArtifacts | FeaturePipelinePlan | ModelBuildPlan:
    validated = validate_dag_json(dag)
    manifest = _build_manifest(validated, path_type)
    if path_type == "dag_only":
        return DagArtifacts(
            node_inventory=[{"node_id": node.node_id, "op_uid": node.op_uid, "kind": node.kind} for node in validated.nodes],
            edge_inventory=[{"source": edge.source, "target": edge.target} for edge in validated.edges],
            method_description="DAG structural prior generated from the canonical operator catalog.",
            manifest=manifest,
        )

    feature_specs = _build_feature_specs(validated)
    if path_type == "ml":
        return FeaturePipelinePlan(feature_specs=feature_specs, manifest=manifest)

    return ModelBuildPlan(
        backend_target="torch",
        feature_specs=feature_specs,
        trainable_head={"input_dim": len(feature_specs), "hidden_dim": 8, "output_dim": 2},
        manifest=manifest,
    )
