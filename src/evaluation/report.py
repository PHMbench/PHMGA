"""Report rendering for graph-dependent artifacts."""

from __future__ import annotations

from typing import Any, Dict

from src.bridge import CompiledDagManifest
from src.data import DatasetProtocol
from src.dag import DagJson
from src.states import WorkflowState


def render_mermaid_dag(dag: DagJson) -> str:
    """Render the validated DAG into a lightweight Mermaid diagram."""
    lines = ["```mermaid", "flowchart TD"]
    for node in dag.nodes:
        lines.append(f"    {node.node_id}[{node.name}]")
    for edge in dag.edges:
        lines.append(f"    {edge.source} --> {edge.target}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def build_final_report(
    state: WorkflowState,
    protocol: DatasetProtocol,
    manifest: CompiledDagManifest,
    path_artifacts: Dict[str, Any],
) -> str:
    """Assemble the final markdown report from protocol and artifact evidence."""
    plan_steps = len(state.step_plan.plan) if state.step_plan else 0
    lines = [
        f"# PHMGA Final Report: {protocol.dataset_name} / {state.graph_path}",
        "",
        "## Conclusion",
        f"- Dataset: {protocol.dataset_name}",
        f"- Graph path: {state.graph_path}",
        f"- DAG hash: `{manifest.dag_hash}`",
        f"- Nodes: {len(manifest.nodes)}",
    ]
    metrics = path_artifacts.get("metrics")
    if isinstance(metrics, dict):
        for split_name in ("train", "val", "test"):
            if split_name in metrics:
                lines.append(
                    f"- {split_name}: acc={metrics[split_name]['accuracy']:.3f}, macro_f1={metrics[split_name]['macro_f1']:.3f}"
                )
    lines.extend(
        [
            "",
            "## Protocol",
            f"- Catalog: {protocol.catalog}",
            f"- Metadata schema: {protocol.metadata_schema_version}",
            f"- Split sizes: train={len(protocol.splits.train_ids)}, val={len(protocol.splits.val_ids)}, test={len(protocol.splits.test_ids)}",
            f"- Window: size={protocol.window.window_size}, stride={protocol.window.stride}, mode={protocol.window.slice_mode}",
            "",
            "## Evidence",
            f"- Manifest warnings: {len(manifest.warnings)}",
            f"- Artifact keys: {', '.join(sorted(path_artifacts.keys()))}",
            "",
            "## Workflow",
            f"- User instruction: {state.user_instruction}",
            f"- Plan steps: {plan_steps}",
            "",
            "## Analysis Workflow",
            "!Analysis Workflow",
        ]
    )
    return "\n".join(lines) + "\n"
