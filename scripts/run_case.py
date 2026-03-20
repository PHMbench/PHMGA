"""Library execution routine invoked by the root Hydra entrypoint."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from src.bridge import DagArtifacts, FeaturePipelinePlan, ModelBuildPlan
from src.data import build_protocol_from_config
from src.evaluation import render_mermaid_dag
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.phm_outer_graph import run_phm_graph
from src.states import WorkflowState
from src.utils import ensure_dir, write_json, write_text


def _record_json_artifact(artifact_index: Dict[str, str], output_dir: Path, filename: str, payload: Any) -> None:
    write_json(payload, output_dir / filename)
    artifact_index[filename] = filename


def _record_text_artifact(artifact_index: Dict[str, str], output_dir: Path, filename: str, content: str) -> None:
    write_text(content, output_dir / filename)
    artifact_index[filename] = filename


def _decision_side_outputs(state: WorkflowState) -> Dict[str, Any]:
    if not state.dag:
        return {}
    outputs: Dict[str, Any] = {}
    for node in state.dag.nodes:
        if node.kind != "decision":
            continue
        if node.node_id in state.execution_results:
            outputs[node.node_id] = state.execution_results[node.node_id]
    return outputs


def _write_common_artifacts(
    output_dir: Path,
    artifact_index: Dict[str, str],
    state: WorkflowState,
    compiled: Any,
    protocol,
) -> None:
    """Write artifacts that every graph path shares."""
    if hasattr(compiled, "manifest"):
        manifest_payload = compiled.manifest.model_dump()
    else:
        manifest_payload = dict(compiled)
    dag_payload = state.dag.model_dump()
    _record_json_artifact(artifact_index, output_dir, "validated_dag.json", dag_payload)
    _record_json_artifact(artifact_index, output_dir, "compiled_dag_manifest.json", manifest_payload)
    _record_text_artifact(artifact_index, output_dir, "dag_graph.md", render_mermaid_dag(state.dag))
    _record_json_artifact(artifact_index, output_dir, "resolved_splits.json", protocol.splits.model_dump())
    _record_json_artifact(
        artifact_index,
        output_dir,
        "resolved_dataset_manifest.json",
        protocol.model_dump(exclude={"splits"}),
    )
    decision_outputs = _decision_side_outputs(state)
    if decision_outputs:
        _record_json_artifact(artifact_index, output_dir, "decision_side_outputs.json", decision_outputs)
    for trace_name in (
        "planner_transport_trace.json",
        "planner_normalization_trace.json",
        "planner_raw_response.txt",
        "planner_repair_response.txt",
    ):
        trace_path = output_dir / trace_name
        if trace_path.exists():
            artifact_index[trace_name] = trace_name


def _run_frontend_loop(
    state: WorkflowState,
    protocol,
    llm,
    catalog,
    runtime_config: Dict[str, Any],
) -> WorkflowState:
    """Compatibility wrapper over the LangGraph frontend runtime."""

    return run_phm_graph(state, protocol, catalog, runtime_config, llm_override=llm)


def run_case(
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the full paper-oriented workflow for one config-defined case."""
    protocol = build_protocol_from_config(runtime_config)
    llm = get_llm(runtime_config)
    catalog = get_operator_catalog()
    graph_path = runtime_config["experiment"]["graph_path"]
    state = WorkflowState(
        user_instruction=str(runtime_config["experiment"]["user_instruction"]),
        dataset_name=protocol.dataset_name,
        graph_path=graph_path,
        runtime_config=runtime_config,
        max_iterations=int(runtime_config.get("runtime", {}).get("max_iterations", 4)),
        data_context={
            "catalog": protocol.catalog,
            "metadata_schema_version": protocol.metadata_schema_version,
            "source_mode": protocol.source_mode,
            "min_depth": 2,
            "min_width": 1,
            "max_depth": 8,
            "stage": "RUN_CASE",
        },
    )
    state = _run_frontend_loop(state, protocol, llm, catalog, runtime_config)
    if state.halt_reason:
        raise RuntimeError(state.halt_reason)
    compiled = state.compiled_bundle if state.compiled_bundle is not None else state.compiled_manifest
    path_artifacts = state.path_artifacts

    output_root = ensure_dir(runtime_config["runtime"]["output_dir"])
    artifact_index: Dict[str, str] = {}
    _write_common_artifacts(output_root, artifact_index, state, compiled, protocol)

    if state.graph_path == "dag_only":
        _record_json_artifact(artifact_index, output_root, "dag_artifacts.json", path_artifacts)
        _record_text_artifact(artifact_index, output_root, "method_description.md", path_artifacts["method_description"])
    elif state.graph_path == "ml":
        _record_json_artifact(artifact_index, output_root, "feature_pipeline.json", path_artifacts["feature_pipeline"])
        _record_json_artifact(artifact_index, output_root, "feature_list.json", path_artifacts["feature_list"])
        _record_json_artifact(
            artifact_index,
            output_root,
            "feature_separability_summary.json",
            path_artifacts["feature_separability_summary"],
        )
        _record_json_artifact(artifact_index, output_root, "metrics.json", path_artifacts["metrics"])
        _record_json_artifact(artifact_index, output_root, "predictions.json", path_artifacts["predictions"])
        _record_json_artifact(artifact_index, output_root, "importance.json", path_artifacts["importance"])
        _record_json_artifact(
            artifact_index,
            output_root,
            "similarity_artifacts.json",
            path_artifacts["similarity_artifacts"],
        )
    else:
        _record_json_artifact(artifact_index, output_root, "model_build_plan.json", path_artifacts["model_build_plan"])
        _record_json_artifact(artifact_index, output_root, "training_curves.json", path_artifacts["training_curves"])
        _record_json_artifact(artifact_index, output_root, "checkpoint.json", path_artifacts["checkpoint"])
        _record_json_artifact(artifact_index, output_root, "importance.json", path_artifacts["importance"])
        _record_json_artifact(artifact_index, output_root, "metrics.json", path_artifacts["metrics"])
        _record_json_artifact(
            artifact_index,
            output_root,
            "similarity_artifacts.json",
            path_artifacts["similarity_artifacts"],
        )
        if "control_statistics" in path_artifacts:
            _record_json_artifact(
                artifact_index,
                output_root,
                "control_statistics.json",
                path_artifacts["control_statistics"],
            )

    _record_text_artifact(artifact_index, output_root, "final_report.md", state.final_report)
    _record_json_artifact(artifact_index, output_root, "resolved_config.json", runtime_config)
    runtime_trace_payload = state.data_context.pop("_dataset_level_runtime_trace", None)
    if state.dag_quality_summary:
        dag_quality_payload = deepcopy(state.dag_quality_summary)
        _record_json_artifact(artifact_index, output_root, "dag_quality_summary.json", dag_quality_payload)
        state.dag_quality_summary = dag_quality_payload
    if runtime_trace_payload is not None:
        _record_json_artifact(
            artifact_index,
            output_root,
            "dataset_level_runtime_trace.json",
            runtime_trace_payload,
        )
    artifact_index["workflow_state.json"] = "workflow_state.json"
    artifact_index["artifact_index.json"] = "artifact_index.json"
    state.artifact_index = dict(artifact_index)
    workflow_state_payload = state.model_dump(
        exclude={
            "artifact_index",
            "execution_results",
            "last_stable_execution_results",
        }
    )
    workflow_state_payload["artifact_index_path"] = "artifact_index.json"
    _record_json_artifact(
        artifact_index,
        output_root,
        "workflow_state.json",
        workflow_state_payload,
    )
    state.artifact_index = dict(artifact_index)
    _record_json_artifact(artifact_index, output_root, "artifact_index.json", artifact_index)

    return {
        "config_name": runtime_config["runtime"]["config_name"],
        "dataset": protocol.dataset_name,
        "graph_path": state.graph_path,
        "source_mode": protocol.source_mode,
        "output_dir": str(output_root),
        "manifest_path": str(output_root / "compiled_dag_manifest.json"),
        "report_path": str(output_root / "final_report.md"),
    }
