"""Single execution entrypoint for all graph paths in the rebuilt repo."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import execute_agent, plan_agent, reflect_agent, report_agent
from src.bridge import DagArtifacts, FeaturePipelinePlan, ModelBuildPlan, compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_split_signals
from src.evaluation import build_dag_quality_summary, render_mermaid_dag
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import RoundTrace, WorkflowState
from src.training import run_ml_pipeline, run_torch_pipeline
from src.utils import ensure_dir, write_json, write_text
from src.utils import hash_payload


def _resolve_output_policy(runtime_config: Dict[str, Any], graph_path: str) -> str:
    if graph_path == "ml":
        return str(runtime_config["model"]["ml"].get("output_policy", "terminal_only"))
    if graph_path == "torch":
        return str(runtime_config["model"]["torch"].get("output_policy", "terminal_only"))
    return "terminal_only"


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
    write_json(state.dag.model_dump(), output_dir / "dag.json")
    write_json(compiled.manifest.model_dump(), output_dir / "compiled_dag_manifest.json")
    write_text(render_mermaid_dag(state.dag), output_dir / "dag_graph.md")
    write_json(protocol.splits.model_dump(), output_dir / "resolved_splits.json")
    write_json(protocol.model_dump(), output_dir / "resolved_dataset_manifest.json")
    decision_outputs = _decision_side_outputs(state)
    if decision_outputs:
        write_json(decision_outputs, output_dir / "decision_side_outputs.json")


def _run_path(
    graph_path: str,
    compiled: Union[DagArtifacts, FeaturePipelinePlan, ModelBuildPlan],
    split_records: Optional[Dict[str, Any]],
    runtime_config: Dict[str, Any],
    catalog,
) -> Dict[str, Any]:
    """Dispatch from one compiled DAG to the selected backend path."""
    if graph_path == "dag_only":
        payload = compiled.model_dump()
        payload["artifact_kind"] = "dag_only"
        return payload
    if split_records is None:
        raise ValueError(f"split_records are required for graph_path={graph_path}")
    if graph_path == "ml":
        return run_ml_pipeline(
            compiled,
            split_records,
            catalog,
            algorithm=str(runtime_config["model"]["ml"].get("algorithm", "logistic_regression")),
            max_iter=int(runtime_config["model"]["ml"]["max_iter"]),
        )
    return run_torch_pipeline(
        compiled,
        split_records,
        catalog,
        epochs=int(runtime_config["model"]["torch"]["epochs"]),
        learning_rate=float(runtime_config["model"]["torch"]["learning_rate"]),
        device=str(runtime_config["model"]["torch"].get("device", "auto")),
        phase=str(runtime_config["model"]["torch"].get("phase", "compiled")),
        module_runtime_enabled=bool(runtime_config["model"]["torch"].get("module_runtime", {}).get("enabled", False)),
        control_default_mode=str(runtime_config["model"]["torch"].get("control", {}).get("default_mode", "fixed")),
        tau=float(runtime_config["model"]["torch"].get("control", {}).get("tau", 1.0)),
        attention_heads=int(runtime_config["model"]["torch"].get("control", {}).get("attention_heads", 1)),
        attention_dropout=float(runtime_config["model"]["torch"].get("control", {}).get("attention_dropout", 0.0)),
    )


def _dag_hash(state: WorkflowState) -> str:
    if state.dag is None:
        return hash_payload({"nodes": [], "edges": []})
    return hash_payload(state.dag.model_dump())


def _run_frontend_loop(
    state: WorkflowState,
    protocol,
    llm,
    catalog,
    runtime_config: Dict[str, Any],
) -> WorkflowState:
    """Execute the front-end agent loop with rollback-aware replan handling."""

    state.last_stable_dag = state.dag.model_copy(deep=True) if state.dag else None
    state.last_stable_execution_results = deepcopy(state.execution_results)

    for round_index in range(1, state.max_iterations + 1):
        state.iteration_index = round_index
        input_dag_hash = _dag_hash(state)
        previous_node_ids = {node.node_id for node in state.dag.nodes} if state.dag else set()

        state = plan_agent(state, protocol, llm, catalog)
        state = execute_agent(state, protocol, catalog, llm)
        if bool(runtime_config.get("evaluation", {}).get("dag_quality", {}).get("enabled", True)):
            state.dag_quality_summary = build_dag_quality_summary(
                state,
                protocol,
                runtime_config,
                catalog,
            ).model_dump()
        else:
            state.dag_quality_summary = {}
        state = reflect_agent(state, llm)

        current_reflection = state.reflection_results[-1]
        current_node_ids = {node.node_id for node in state.dag.nodes} if state.dag else set()
        added_node_ids = sorted(current_node_ids - previous_node_ids)
        round_step_plan = state.step_plan.model_copy(deep=True) if state.step_plan else None

        rolled_back = False
        if current_reflection.decision == "need_replan":
            rolled_back = True
            state.dag = state.last_stable_dag.model_copy(deep=True) if state.last_stable_dag else None
            state.execution_results = deepcopy(state.last_stable_execution_results)
            state.step_plan = None

        state.round_history.append(
            RoundTrace(
                round_index=round_index,
                input_dag_hash=input_dag_hash,
                step_plan=round_step_plan,
                added_node_ids=added_node_ids,
                execution_gaps=[gap.model_copy(deep=True) for gap in state.execution_gaps],
                reflection_result=current_reflection.model_copy(deep=True),
                rolled_back=rolled_back,
            )
        )

        if current_reflection.decision == "need_replan":
            continue
        if current_reflection.decision == "need_patch":
            state.last_stable_dag = state.dag.model_copy(deep=True) if state.dag else None
            state.last_stable_execution_results = deepcopy(state.execution_results)
            continue
        if current_reflection.decision == "finish":
            state.last_stable_dag = state.dag.model_copy(deep=True) if state.dag else None
            state.last_stable_execution_results = deepcopy(state.execution_results)
            return state
        raise RuntimeError(f"Workflow halted: {current_reflection.reason}")

    raise RuntimeError(
        f"Workflow exceeded max_iterations={state.max_iterations} without reaching finish."
    )


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
    # The validated DAG JSON is the only legal hand-off into backend execution.
    compiled = compile_dag_for_path(
        state.dag,
        state.graph_path,
        output_policy=_resolve_output_policy(runtime_config, state.graph_path),
    )
    split_records = None if state.graph_path == "dag_only" else materialize_split_signals(protocol)
    path_artifacts = _run_path(state.graph_path, compiled, split_records, runtime_config, catalog)
    decision_outputs = _decision_side_outputs(state)
    if decision_outputs:
        path_artifacts["decision_side_outputs"] = decision_outputs

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

    final_report = report_agent(state, protocol, compiled.manifest, path_artifacts, llm)
    write_text(final_report, output_root / "final_report.md")
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
