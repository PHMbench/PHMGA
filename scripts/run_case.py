from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import execute_agent, plan_agent, reflect_agent, report_agent
from src.bridge import DagArtifacts, FeaturePipelinePlan, ModelBuildPlan, compile_dag_for_path
from src.config import load_runtime_config
from src.data import build_protocol_from_config, materialize_split_signals
from src.evaluation import render_mermaid_dag
from src.llm import get_llm
from src.operators import get_operator_catalog
from src.states import WorkflowState
from src.training import run_ml_pipeline, run_torch_pipeline
from src.utils import ensure_dir, write_json, write_text


def _write_common_artifacts(output_dir: Path, state: WorkflowState, compiled: Any) -> None:
    write_json(state.dag.model_dump(), output_dir / "dag.json")
    write_json(compiled.manifest.model_dump(), output_dir / "compiled_dag_manifest.json")
    write_text(render_mermaid_dag(state.dag), output_dir / "dag_graph.md")


def _run_path(
    graph_path: str,
    compiled: DagArtifacts | FeaturePipelinePlan | ModelBuildPlan,
    split_records: Dict[str, Any],
    runtime_config: Dict[str, Any],
    catalog,
) -> Dict[str, Any]:
    if graph_path == "dag_only":
        payload = compiled.model_dump()
        payload["artifact_kind"] = "dag_only"
        return payload
    if graph_path == "ml":
        return run_ml_pipeline(
            compiled,
            split_records,
            catalog,
            max_iter=int(runtime_config["model"]["ml"]["max_iter"]),
        )
    return run_torch_pipeline(
        compiled,
        split_records,
        catalog,
        epochs=int(runtime_config["model"]["torch"]["epochs"]),
        learning_rate=float(runtime_config["model"]["torch"]["learning_rate"]),
    )


def run_case(
    config_path: str,
    *,
    dataset_name: str | None = None,
    graph_path: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    runtime_config = load_runtime_config(
        config_path,
        dataset_name=dataset_name,
        graph_path=graph_path,
        output_dir=output_dir,
    )
    protocol = build_protocol_from_config(runtime_config)
    llm = get_llm(runtime_config)
    catalog = get_operator_catalog()
    state = WorkflowState(
        user_instruction="Generate a paper-ready PHM workflow from canonical metadata.",
        dataset_name=protocol.dataset_name,
        graph_path=runtime_config["experiment"]["graph_path"],
        data_context={"catalog": protocol.catalog, "metadata_schema_version": protocol.metadata_schema_version},
    )
    state = plan_agent(state, protocol, llm)
    state = execute_agent(state, protocol, catalog)
    state = reflect_agent(state, llm)
    compiled = compile_dag_for_path(state.dag, state.graph_path)
    split_records = materialize_split_signals(protocol)
    path_artifacts = _run_path(state.graph_path, compiled, split_records, runtime_config, catalog)

    output_root = ensure_dir(runtime_config["runtime"]["output_dir"])
    _write_common_artifacts(output_root, state, compiled)

    if state.graph_path == "dag_only":
        write_json(path_artifacts, output_root / "dag_artifacts.json")
        write_text(path_artifacts["method_description"], output_root / "method_description.md")
    elif state.graph_path == "ml":
        write_json(path_artifacts["feature_pipeline"], output_root / "feature_pipeline.json")
        write_json(path_artifacts["metrics"], output_root / "metrics.json")
        write_json(path_artifacts["predictions"], output_root / "predictions.json")
        write_json(path_artifacts["importance"], output_root / "importance.json")
    else:
        write_json(path_artifacts["model_build_plan"], output_root / "model_build_plan.json")
        write_json(path_artifacts["training_curves"], output_root / "training_curves.json")
        write_json(path_artifacts["checkpoint"], output_root / "checkpoint.json")
        write_json(path_artifacts["importance"], output_root / "importance.json")
        write_json(path_artifacts["metrics"], output_root / "metrics.json")

    final_report = report_agent(state, protocol, compiled.manifest, path_artifacts)
    write_text(final_report, output_root / "final_report.md")
    write_json(runtime_config, output_root / "resolved_config.json")

    return {
        "dataset": protocol.dataset_name,
        "graph_path": state.graph_path,
        "output_dir": str(output_root),
        "manifest_path": str(output_root / "compiled_dag_manifest.json"),
        "report_path": str(output_root / "final_report.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    result = run_case(
        args.config,
        dataset_name=args.dataset,
        graph_path=args.graph_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
