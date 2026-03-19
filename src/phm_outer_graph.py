"""LangGraph frontend runtime for the paper-oriented PHM workflow."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Union

from langgraph.graph import END, START, StateGraph

from src.agents import execute_agent, inquirer_agent, plan_agent, reflect_agent, report_agent
from src.bridge import CompiledDagManifest, DagArtifacts, FeaturePipelinePlan, ModelBuildPlan, compile_dag_for_path
from src.data import materialize_split_signals
from src.evaluation import build_dag_quality_summary
from src.operators import OperatorCatalog
from src.states import PHMState, RoundTrace
from src.training import run_ml_pipeline, run_torch_pipeline
from src.utils import hash_payload


def _resolve_output_policy(runtime_config: Dict[str, Any], graph_path: str) -> str:
    if graph_path == "ml":
        return str(runtime_config["model"]["ml"].get("output_policy", "terminal_only"))
    if graph_path == "torch":
        return str(runtime_config["model"]["torch"].get("output_policy", "terminal_only"))
    return "terminal_only"


def _decision_side_outputs(state: PHMState) -> Dict[str, Any]:
    if not state.dag:
        return {}
    outputs: Dict[str, Any] = {}
    for node in state.dag.nodes:
        if node.kind != "decision":
            continue
        if node.node_id in state.execution_results:
            outputs[node.node_id] = state.execution_results[node.node_id]
    return outputs


def _run_path(
    graph_path: str,
    compiled: Union[DagArtifacts, FeaturePipelinePlan, ModelBuildPlan],
    split_records: Optional[Dict[str, Any]],
    runtime_config: Dict[str, Any],
    catalog: OperatorCatalog,
    dataset_name: str,
) -> Dict[str, Any]:
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
            dataset_name=dataset_name,
            backend_provider=str(runtime_config.get("llm", {}).get("provider", "unknown")),
            backend_model=str(runtime_config.get("llm", {}).get("model", "unknown")),
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


def _dag_hash(state: PHMState) -> str:
    if state.dag is None:
        return hash_payload({"nodes": [], "edges": []})
    return hash_payload(state.dag.model_dump())


def build_phm_graph(protocol, catalog: OperatorCatalog, runtime_config: Dict[str, Any], llm_override=None):
    builder = StateGraph(PHMState)

    def plan_node(state: PHMState) -> Dict[str, Any]:
        state.iteration_index += 1
        state.current_round_input_hash = _dag_hash(state)
        state.current_round_previous_node_ids = [node.node_id for node in state.dag.nodes] if state.dag else []
        state = plan_agent(state, protocol, llm_override, catalog)
        return state.model_dump()

    def execute_node(state: PHMState) -> Dict[str, Any]:
        state = execute_agent(state, protocol, catalog, llm_override)
        return state.model_dump()

    def dag_quality_node(state: PHMState) -> Dict[str, Any]:
        if bool(runtime_config.get("evaluation", {}).get("dag_quality", {}).get("enabled", True)):
            state.dag_quality_summary = build_dag_quality_summary(state, protocol, runtime_config, catalog).model_dump()
        else:
            state.dag_quality_summary = {}
        state.status = "dag_quality_evaluated"
        return state.model_dump()

    def reflect_node(state: PHMState) -> Dict[str, Any]:
        state = reflect_agent(state, llm_override)
        current_reflection = state.reflection_results[-1]
        current_node_ids = {node.node_id for node in state.dag.nodes} if state.dag else set()
        added_node_ids = sorted(current_node_ids - set(state.current_round_previous_node_ids))
        state.round_history.append(
            RoundTrace(
                round_index=state.iteration_index,
                input_dag_hash=state.current_round_input_hash,
                step_plan=state.step_plan.model_copy(deep=True) if state.step_plan else None,
                added_node_ids=added_node_ids,
                execution_gaps=[gap.model_copy(deep=True) for gap in state.execution_gaps],
                reflection_result=current_reflection.model_copy(deep=True),
                rolled_back=current_reflection.decision == "need_replan",
            )
        )
        if current_reflection.decision in {"need_patch", "finish"}:
            state.stable_snapshot()
        elif current_reflection.decision == "halt":
            state.halt_reason = current_reflection.reason
        return state.model_dump()

    def rollback_node(state: PHMState) -> Dict[str, Any]:
        state.dag = state.last_stable_dag.model_copy(deep=True) if state.last_stable_dag else None
        state.execution_results = deepcopy(state.last_stable_execution_results)
        state.step_plan = None
        state.status = "rolled_back"
        return state.model_dump()

    def compile_ready_node(state: PHMState) -> Dict[str, Any]:
        compiled = compile_dag_for_path(
            state.dag,
            state.graph_path,
            output_policy=_resolve_output_policy(runtime_config, state.graph_path),
        )
        split_records = None if state.graph_path == "dag_only" else materialize_split_signals(protocol)
        path_artifacts = _run_path(state.graph_path, compiled, split_records, runtime_config, catalog, protocol.dataset_name)
        decision_outputs = _decision_side_outputs(state)
        if decision_outputs:
            path_artifacts["decision_side_outputs"] = decision_outputs
        state.compiled_bundle = compiled
        state.compiled_manifest = compiled.manifest.model_dump()
        state.path_artifacts = path_artifacts
        state.status = "compiled"
        return state.model_dump()

    def inquirer_node(state: PHMState) -> Dict[str, Any]:
        state = inquirer_agent(state)
        return state.model_dump()

    def report_node(state: PHMState) -> Dict[str, Any]:
        state.final_report = report_agent(
            state,
            protocol,
            CompiledDagManifest.model_validate(state.compiled_manifest),
            state.path_artifacts,
            llm_override,
        )
        state.status = "reported"
        return state.model_dump()

    def halt_node(state: PHMState) -> Dict[str, Any]:
        if not state.halt_reason:
            state.halt_reason = "Workflow halted."
        state.status = "halted"
        return state.model_dump()

    def reflect_router(state: PHMState) -> str:
        decision = state.reflection_results[-1].decision
        if decision == "finish":
            return "compile_ready"
        if decision == "need_patch":
            if state.iteration_index >= state.max_iterations:
                state.halt_reason = f"Workflow exceeded max_iterations={state.max_iterations} without reaching finish."
                return "halt"
            return "plan"
        if decision == "need_replan":
            if state.iteration_index >= state.max_iterations:
                state.halt_reason = f"Workflow exceeded max_iterations={state.max_iterations} without reaching finish."
                return "halt"
            return "rollback"
        return "halt"

    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("dag_quality", dag_quality_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("rollback", rollback_node)
    builder.add_node("compile_ready", compile_ready_node)
    builder.add_node("inquirer", inquirer_node)
    builder.add_node("report", report_node)
    builder.add_node("halt", halt_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "dag_quality")
    builder.add_edge("dag_quality", "reflect")
    builder.add_conditional_edges(
        "reflect",
        reflect_router,
        {
            "plan": "plan",
            "rollback": "rollback",
            "compile_ready": "compile_ready",
            "halt": "halt",
        },
    )
    builder.add_edge("rollback", "plan")
    builder.add_edge("compile_ready", "inquirer")
    builder.add_edge("inquirer", "report")
    builder.add_edge("report", END)
    builder.add_edge("halt", END)
    return builder.compile()


def run_phm_graph(
    state: PHMState,
    protocol,
    catalog: OperatorCatalog,
    runtime_config: Dict[str, Any],
    llm_override=None,
) -> PHMState:
    graph = build_phm_graph(protocol, catalog, runtime_config, llm_override)
    return PHMState.model_validate(graph.invoke(state))
