"""Execute agent that materializes a structured plan into DAG nodes.

The executor remains strictly plan-driven. It uses richer operator schema
metadata to validate input arity, rank behavior, and parameter resolution
before adding each node to the validated DAG candidate.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.dag import DAGTracker, DagNode
from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import OfflineLLM
from src.operators import OperatorCatalog
from src.prompts import render_execute_prompt
from src.states import ExecutionGap, WorkflowState


def _node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{parent.replace(',', '__')}"


def _node_kind(schema_category: str) -> str:
    if schema_category == "AGGREGATE":
        return "feature"
    if schema_category == "MULTI_VARIABLE":
        return "multi"
    if schema_category == "DECISION":
        return "decision"
    return "transform"


def _validate_parent_contract(operator, parent_results: List[np.ndarray]) -> str | None:
    expected_arity = operator.spec.input_spec.get("arity")
    if operator.spec.rank_class == "multi_input" or expected_arity == "multi":
        if len(parent_results) < int(operator.spec.input_spec.get("min_parents", 2)):
            return "Multi-input operator requires at least two parent nodes."
    elif len(parent_results) != 1:
        return "Single-input operator received multiple parents."

    min_rank = operator.spec.input_spec.get("min_rank")
    if min_rank is not None:
        for parent_result in parent_results:
            parent_rank = int(np.asarray(parent_result).ndim)
            if parent_rank < int(min_rank):
                return (
                    f"Parent rank {parent_rank} violates min_rank={int(min_rank)} "
                    f"for op '{operator.spec.op_name}'."
                )
    return None


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
                operator_category="INPUT",
                rank_class="rank_same",
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


def _execute_single(op, parent_result: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_result, **params), dtype=float)


def _execute_multi(op, parent_results: List[np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_results, **params), dtype=float)


def _execute_decision(op, parent_result: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(op.forward_np(parent_result, **params))


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

        parent_results = [np.asarray(state.execution_results[parent_id], dtype=float) for parent_id in parent_ids]
        contract_error = _validate_parent_contract(operator, parent_results)
        if contract_error:
            state.execution_gaps.append(
                ExecutionGap(
                    step_index=step_index,
                    parent=step.parent,
                    op_name=step.op_name,
                    message=contract_error,
                    recoverable=False,
                )
            )
            continue

        parent_summaries = [
            {
                "node_id": parent_id,
                "shape": list(parent_result.shape),
            }
            for parent_id, parent_result in zip(parent_ids, parent_results)
        ]

        try:
            params = llm.resolve_missing_params(
                op_name=step.op_name,
                param_schema=operator.spec.param_schema,
                param_defaults=operator.spec.param_defaults,
                param_docs=operator.spec.param_docs,
                llm_tunable_params=operator.spec.llm_tunable_params,
                provided_params=step.params,
                signal_context=state.signal_context,
                parent_summaries=parent_summaries,
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

        if operator.spec.schema_category == "DECISION":
            result = _execute_decision(operator, parent_results[0], params)
            input_bindings = {}
            in_shape = list(parent_results[0].shape)
            out_shape = [1]
        elif operator.spec.schema_category == "MULTI_VARIABLE":
            result = _execute_multi(operator, parent_results, params)
            input_bindings = {f"arg{index}": parent_id for index, parent_id in enumerate(parent_ids)}
            in_shape = [sum(int(np.asarray(result_item).size) for result_item in parent_results)]
            out_shape = list(np.asarray(result).shape) or [1]
        else:
            result = _execute_single(operator, parent_results[0], params)
            input_bindings = {}
            in_shape = list(parent_results[0].shape)
            out_shape = list(np.asarray(result).shape) or [1]

        new_node_id = _node_id(step_index, step.op_name, step.parent)
        tracker.add_node(
            DagNode(
                node_id=new_node_id,
                op_uid=operator.spec.op_uid,
                name=operator.spec.name,
                kind=_node_kind(operator.spec.schema_category),
                operator_category=operator.spec.schema_category,
                rank_class=operator.spec.rank_class,
                params=params,
                parents=parent_ids,
                in_shape=in_shape,
                out_shape=out_shape,
                backend_availability=operator.spec.backend_availability,
                execution_role=operator.spec.execution_role,
                legal_paths=operator.spec.legal_paths,
                input_bindings=input_bindings,
                plan_step_ref=f"round_{state.iteration_index:02d}_step_{step_index:02d}",
                rationale=(
                    f"Round {state.iteration_index} planner step {step_index}: "
                    f"apply {step.op_name} to {step.parent} using schema category "
                    f"{operator.spec.schema_category} and rank class {operator.spec.rank_class}."
                ),
            )
        )
        state.execution_results[new_node_id] = result

    state.dag = tracker.export()
    state.status = "executed"
    return state
