"""Single execution entrypoint for all graph paths in the rebuilt repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bridge import DagArtifacts, FeaturePipelinePlan, ModelBuildPlan
from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.evaluation import render_mermaid_dag
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.phm_outer_graph import run_phm_graph
from src.states import WorkflowState
from src.utils import ensure_dir, write_json, write_text


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


def _write_common_artifacts(output_dir: Path, state: WorkflowState, compiled: Any, protocol) -> None:
    """Write artifacts that every graph path shares."""
    if hasattr(compiled, "manifest"):
        manifest_payload = compiled.manifest.model_dump()
    else:
        manifest_payload = dict(compiled)
    write_json(state.dag.model_dump(), output_dir / "dag.json")
    write_json(manifest_payload, output_dir / "compiled_dag_manifest.json")
    write_text(render_mermaid_dag(state.dag), output_dir / "dag_graph.md")
    write_json(protocol.splits.model_dump(), output_dir / "resolved_splits.json")
    write_json(protocol.model_dump(), output_dir / "resolved_dataset_manifest.json")
    decision_outputs = _decision_side_outputs(state)
    if decision_outputs:
        write_json(decision_outputs, output_dir / "decision_side_outputs.json")


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
    config_input: Union[str, Path, Dict[str, Any]],
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full paper-oriented workflow for one config-defined case."""
    runtime_config = load_runtime_config(config_input, output_dir=output_dir)
    protocol = build_protocol_from_config(runtime_config)
    llm = get_llm(runtime_config)
    catalog = get_operator_catalog()
    graph_path = runtime_config["experiment"]["graph_path"]
    state = WorkflowState(
        user_instruction="Generate a paper-ready PHM workflow from canonical metadata.",
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
    _write_common_artifacts(output_root, state, compiled, protocol)

    if state.graph_path == "dag_only":
        write_json(path_artifacts, output_root / "dag_artifacts.json")
        write_text(path_artifacts["method_description"], output_root / "method_description.md")
    elif state.graph_path == "ml":
        write_json(path_artifacts["feature_pipeline"], output_root / "feature_pipeline.json")
        write_json(path_artifacts["metrics"], output_root / "metrics.json")
        write_json(path_artifacts["predictions"], output_root / "predictions.json")
        write_json(path_artifacts["importance"], output_root / "importance.json")
        write_json(path_artifacts["similarity_artifacts"], output_root / "similarity_artifacts.json")
    else:
        write_json(path_artifacts["model_build_plan"], output_root / "model_build_plan.json")
        write_json(path_artifacts["training_curves"], output_root / "training_curves.json")
        write_json(path_artifacts["checkpoint"], output_root / "checkpoint.json")
        write_json(path_artifacts["importance"], output_root / "importance.json")
        write_json(path_artifacts["metrics"], output_root / "metrics.json")
        write_json(path_artifacts["similarity_artifacts"], output_root / "similarity_artifacts.json")
        if "control_statistics" in path_artifacts:
            write_json(path_artifacts["control_statistics"], output_root / "control_statistics.json")

    write_text(state.final_report, output_root / "final_report.md")
    write_json(runtime_config, output_root / "resolved_config.json")
    write_json(
        state.model_dump(
            exclude={
                "execution_results",
                "last_stable_execution_results",
            }
        ),
        output_root / "workflow_state.json",
    )
    if state.dag_quality_summary:
        write_json(state.dag_quality_summary, output_root / "dag_quality_summary.json")

    return {
        "config_name": runtime_config["runtime"]["config_name"],
        "dataset": protocol.dataset_name,
        "graph_path": state.graph_path,
        "source_mode": protocol.source_mode,
        "output_dir": str(output_root),
        "manifest_path": str(output_root / "compiled_dag_manifest.json"),
        "report_path": str(output_root / "final_report.md"),
    }


def main() -> None:
    """CLI entrypoint for one graph-dependent run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_case(
        args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
