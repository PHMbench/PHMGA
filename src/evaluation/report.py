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
        f"- Operator categories: {', '.join(sorted({node.operator_category for node in manifest.nodes}))}",
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
            f"- Workflow rounds: {len(state.round_history)}",
            f"- Decision side outputs: {sum(1 for node in manifest.nodes if node.kind == 'decision')}",
            "",
            "## DAG Quality",
            f"- Recommendation: {state.dag_quality_summary.get('recommendation_hint', 'n/a')}",
            f"- Quality issues: {', '.join(state.dag_quality_summary.get('issues', []))}",
        ]
    )
    dataset_level = state.dag_quality_summary.get("dataset_level", {})
    if isinstance(dataset_level, dict) and dataset_level:
        lines.extend(
            [
                "",
                "## Dataset-Level Diagnosis Evidence",
                f"- Source: {dataset_level.get('source', 'n/a')}",
                f"- Evidence path: {dataset_level.get('evidence_path', 'n/a')}",
                f"- Materialization ok: {dataset_level.get('materialization_ok', 'n/a')}",
                f"- Distinguishable: {dataset_level.get('distinguishable', 'n/a')}",
                f"- Split window counts: {dataset_level.get('split_window_counts', {})}",
                f"- Feature dims: {dataset_level.get('feature_dims', {})}",
                f"- Dataset-level issues: {', '.join(dataset_level.get('issues', [])) or 'none'}",
            ]
        )
    lines.extend(
        [
            "",
            "## Analysis Workflow",
            "- Front-end main chain: signal_context -> StepPlan -> execute -> dag_quality_evaluator -> reflect",
            "- Replan policy: `need_patch` keeps the current round; `need_replan` rolls back to the last stable DAG.",
            "- Back-end hand-off: validated DAG JSON -> bridge -> graph-dependent artifacts -> final report",
        ]
    )
    if "similarity_artifacts" in path_artifacts:
        lines.extend(
            [
                "",
                "## Optional Similarity Artifacts",
                f"- Keys: {', '.join(sorted(path_artifacts['similarity_artifacts'].keys()))}",
            ]
        )
    return "\n".join(lines) + "\n"
