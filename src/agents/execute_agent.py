"""Execute agent that materializes a structured plan into DAG nodes.

Unlike the old fixed-template executor, this agent only acts on the planner's
`StepPlan`. It also stores representative execution results in workflow state so
reflection and reporting can inspect what was actually materialized.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.dag import DAGTracker, DagNode
from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import OfflineLLM
from src.operators import OperatorCatalog
from src.prompts import render_execute_prompt
from src.states import ExecutionGap, WorkflowState


def _node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{parent.replace(',', '__')}"


def _node_kind(op_uid: str) -> str:
    if op_uid.startswith("feature."):
        return "feature"
    if op_uid.startswith("multi."):
        return "multi"
    if op_uid.startswith("decision."):
        return "decision"
    if op_uid.startswith("input."):
        return "input"
    return "transform"


def _legal_paths(op_uid: str) -> List[str]:
    if op_uid in {"feature.mean", "feature.std"}:
        return ["dag_only", "ml"]
    if op_uid.startswith("decision."):
        return ["dag_only"]
    return ["dag_only", "ml", "torch"]


def _seed_input_roots(state: WorkflowState, protocol: DatasetProtocol, tracker: DAGTracker) -> None:
    if state.signal_context is None:
        raise ValueError("signal_context must be initialized before execute_agent runs.")
    sample_id, preview_window = materialize_preview_signal(protocol)
    state.signal_context.representative_sample_id = sample_id
    for channel_index, root_id in enumerate(state.signal_context.root_node_ids):
        if root_id in state.execution_results:
            continue
        channel_signal = np.asarray(preview_window[channel_index : channel_index + 1], dtype=float)
        tracker.add_node(
            DagNode(
                node_id=root_id,
                op_uid="input.signal",
                name=f"Input Channel {channel_index + 1}",
                kind="input",
                operator_category="input",
                params={"channel_index": channel_index},
                parents=[],
                in_shape=list(channel_signal.shape),
                out_shape=list(channel_signal.shape),
                backend_availability=["np", "pt", "sym"],
                execution_role="fixed",
                legal_paths=["dag_only", "ml", "torch"],
                plan_step_ref="root_input",
                rationale="Implicit multi-channel raw signal root.",
            )
        )
        state.execution_results[root_id] = channel_signal


def _existing_nodes(state: WorkflowState, tracker: DAGTracker) -> None:
    if not state.dag:
        return
    for node in state.dag.nodes:
        tracker.add_node(node)


def _execute_single(op, parent_result: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_result, **params), dtype=float)


def _execute_multi(op, parent_results: List[np.ndarray], params: Dict[str, float]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_results, **params), dtype=float)


def execute_agent(
    state: WorkflowState,
    protocol: DatasetProtocol,
    catalog: OperatorCatalog,
    llm: OfflineLLM,
) -> WorkflowState:
    """Execute the planner output step by step and persist results in state."""

    if state.step_plan is None:
        raise ValueError("execute_agent requires state.step_plan.")
    if state.signal_context is None:
        raise ValueError("execute_agent requires state.signal_context.")

    render_execute_prompt(
        step_plan=state.step_plan.model_dump(),
        dag_json=state.dag.model_dump() if state.dag else None,
        operator_catalog=catalog.summary(),
        signal_context=state.signal_context.model_dump(),
        graph_path=state.graph_path,
    )

    tracker = DAGTracker()
    _existing_nodes(state, tracker)
    _seed_input_roots(state, protocol, tracker)
    state.execution_gaps = []

    for step_index, step in enumerate(state.step_plan.plan, start=1):
        parent_ids = [item.strip() for item in step.parent.split(",") if item.strip()]
        missing_parents = [parent_id for parent_id in parent_ids if parent_id not in state.execution_results]
        if missing_parents:
            state.execution_gaps.append(
                ExecutionGap(
                    step_index=step_index,
                    parent=step.parent,
                    op_name=step.op_name,
                    message=f"Missing parent nodes: {missing_parents}",
                    recoverable=False,
                )
            )
            continue

        try:
            operator = catalog.get_by_plan_name(step.op_name)
        except KeyError as exc:
            state.execution_gaps.append(
                ExecutionGap(
                    step_index=step_index,
                    parent=step.parent,
                    op_name=step.op_name,
                    message=f"Unknown or unsupported operator: {exc}",
                    recoverable=False,
                )
            )
            continue

        try:
            params = llm.resolve_missing_params(
                op_name=step.op_name,
                param_schema=operator.spec.param_schema,
                provided_params=step.params,
                signal_context=state.signal_context,
            )
        except ValueError as exc:
            state.execution_gaps.append(
                ExecutionGap(
                    step_index=step_index,
                    parent=step.parent,
                    op_name=step.op_name,
                    message=str(exc),
                    recoverable=False,
                )
            )
            continue

        parent_results = [np.asarray(state.execution_results[parent_id], dtype=float) for parent_id in parent_ids]
        if operator.spec.op_uid.startswith("decision."):
            state.execution_gaps.append(
                ExecutionGap(
                    step_index=step_index,
                    parent=step.parent,
                    op_name=step.op_name,
                    message="Decision operators are auxiliary terminals and are not executed in the current runtime.",
                    recoverable=True,
                )
            )
            continue

        if operator.spec.op_uid.startswith("multi."):
            result = _execute_multi(operator, parent_results, params)
            input_bindings = {f"arg{index}": parent_id for index, parent_id in enumerate(parent_ids)}
            in_shape = [sum(int(np.asarray(result_item).size) for result_item in parent_results)]
        else:
            if len(parent_results) != 1:
                state.execution_gaps.append(
                    ExecutionGap(
                        step_index=step_index,
                        parent=step.parent,
                        op_name=step.op_name,
                        message="Single-input operator received multiple parents.",
                        recoverable=False,
                    )
                )
                continue
            result = _execute_single(operator, parent_results[0], params)
            input_bindings = {}
            in_shape = list(parent_results[0].shape)

        new_node_id = _node_id(step_index, step.op_name, step.parent)
        tracker.add_node(
            DagNode(
                node_id=new_node_id,
                op_uid=operator.spec.op_uid,
                name=operator.spec.name,
                kind=_node_kind(operator.spec.op_uid),
                operator_category=_node_kind(operator.spec.op_uid),
                params=params,
                parents=parent_ids,
                in_shape=in_shape,
                out_shape=list(np.asarray(result).shape) or [1],
                backend_availability=operator.spec.backend_availability,
                execution_role=operator.spec.execution_role,
                legal_paths=_legal_paths(operator.spec.op_uid),
                input_bindings=input_bindings,
                plan_step_ref=f"step_{step_index:02d}",
                rationale=f"Planner step {step_index}: apply {step.op_name} to {step.parent}.",
            )
        )
        state.execution_results[new_node_id] = result

    state.dag = tracker.export()
    state.status = "executed"
    return state
