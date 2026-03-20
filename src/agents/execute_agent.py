"""Execute agent that materializes a structured plan into DAG nodes.

The executor remains strictly plan-driven. It uses richer operator schema
metadata to validate input arity, rank behavior, and parameter resolution
before adding each node to the validated DAG candidate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np

from src.configuration import Configuration
from src.dag import DAGTracker, DagNode
from src.data import DatasetProtocol, materialize_preview_signal
from src.llm import LLMClient
from src.llm import get_llm as get_llm_client
from src.llm.structured import _local_param_resolution
from src.operators import OperatorCatalog
from src.prompts import render_param_resolution_prompt
from src.states import ExecutionGap, PHMState


def _parent_slug(parent: str) -> str:
    return parent.replace(",", "__")


def _node_index(op_name: str, parent: str, existing_node_ids: List[str]) -> int:
    op_slug = op_name.lower()
    parent_slug = _parent_slug(parent)
    pattern = re.compile(rf"^{re.escape(op_slug)}_(\d+?)_{re.escape(parent_slug)}$")
    return sum(1 for node_id in existing_node_ids if pattern.match(node_id)) + 1


def _canonical_node_id(step_index: int, op_name: str, parent: str) -> str:
    return _legacy_node_id(step_index, op_name, parent)


def _legacy_node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{_parent_slug(parent)}"


def _root_slug(parent: str) -> str:
    roots = re.findall(r"ch\d+", parent)
    deduped: List[str] = []
    for root in roots:
        if root not in deduped:
            deduped.append(root)
    return "__".join(deduped) if deduped else _parent_slug(parent)


def _alias_node_ids(step_index: int, op_name: str, local_index: int, parent: str) -> List[str]:
    op_slug = op_name.lower()
    root_slug = _root_slug(parent)
    aliases = {
        _canonical_node_id(step_index, op_name, parent),
        _legacy_node_id(step_index, op_name, parent),
        f"{op_slug}_{local_index:02d}_{_parent_slug(parent)}",
        f"{op_slug}_{local_index:02d}_{root_slug}",
        f"{op_slug}_{step_index:02d}_{root_slug}",
    }
    return sorted(alias for alias in aliases if alias)


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


def _seed_input_roots(state: PHMState, protocol: DatasetProtocol, tracker: DAGTracker) -> None:
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


def _existing_nodes(state: PHMState, tracker: DAGTracker) -> None:
    if not state.dag:
        return
    for node in state.dag.nodes:
        tracker.add_node(node)


def _resolve_llm(state: PHMState, llm: LLMClient | None) -> LLMClient:
    if llm is not None:
        return llm
    return get_llm_client(Configuration.from_runtime_config(state.runtime_config).to_runtime_dict())


def _execute_single(op, parent_result: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_result, **params), dtype=float)


def _execute_multi(op, parent_results: List[np.ndarray], params: Dict[str, Any]) -> np.ndarray:
    return np.asarray(op.forward_np(parent_results, **params), dtype=float)


def _execute_decision(op, parent_result: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(op.forward_np(parent_result, **params))


def execute_agent(
    state: PHMState,
    protocol: DatasetProtocol,
    catalog: OperatorCatalog,
    llm: LLMClient | None = None,
) -> PHMState:
    """Execute the planner output step by step and persist results in state."""

    if state.step_plan is None:
        raise ValueError("execute_agent requires state.step_plan.")
    if state.signal_context is None:
        raise ValueError("execute_agent requires state.signal_context.")

    tracker = DAGTracker()
    _existing_nodes(state, tracker)
    _seed_input_roots(state, protocol, tracker)
    existing_node_ids = [node.node_id for node in tracker.export().nodes]
    alias_to_canonical: Dict[str, str] = {node_id: node_id for node_id in existing_node_ids}
    state.execution_gaps = []
    llm_client = _resolve_llm(state, llm)

    for step_index, step in enumerate(state.step_plan.plan, start=1):
        raw_parent_ids = [item.strip() for item in step.parent.split(",") if item.strip()]
        parent_ids = [alias_to_canonical.get(parent_id, parent_id) for parent_id in raw_parent_ids]
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
            provided_params = dict(step.params)
            locally_resolved = _local_param_resolution(
                param_schema=operator.spec.param_schema,
                param_defaults=operator.spec.param_defaults,
                provided_params=provided_params,
                signal_context=state.signal_context,
                parent_summaries=parent_summaries,
            )
            missing_params = [
                param_name for param_name in operator.spec.param_schema if param_name not in locally_resolved
            ]
            missing_non_tunable = [
                param_name for param_name in missing_params if param_name not in operator.spec.llm_tunable_params
            ]
            if missing_non_tunable:
                raise ValueError(f"Missing required parameter(s) {missing_non_tunable} for op '{step.op_name}'.")

            unresolved_tunable = [
                param_name for param_name in missing_params if param_name in operator.spec.llm_tunable_params
            ]
            if not unresolved_tunable:
                params = locally_resolved
            else:
                prompt = render_param_resolution_prompt(
                    op_name=step.op_name,
                    requested_params=unresolved_tunable,
                    param_schema=operator.spec.param_schema,
                    param_defaults=operator.spec.param_defaults,
                    param_docs=operator.spec.param_docs,
                    llm_tunable_params=operator.spec.llm_tunable_params,
                    provided_params=provided_params,
                    signal_context=state.signal_context.model_dump(),
                    parent_summaries=parent_summaries,
                )
                params = llm_client.resolve_missing_params(
                    prompt=prompt,
                    op_name=step.op_name,
                    param_schema=operator.spec.param_schema,
                    param_defaults=operator.spec.param_defaults,
                    param_docs=operator.spec.param_docs,
                    llm_tunable_params=operator.spec.llm_tunable_params,
                    provided_params=provided_params,
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

        local_index = _node_index(step.op_name, step.parent, existing_node_ids)
        new_node_id = _canonical_node_id(step_index, step.op_name, step.parent)
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
        existing_node_ids.append(new_node_id)
        for alias_id in _alias_node_ids(step_index, step.op_name, local_index, step.parent):
            alias_to_canonical[alias_id] = new_node_id
            state.execution_results.setdefault(alias_id, result)

    state.dag = tracker.export()
    state.status = "executed"
    return state
