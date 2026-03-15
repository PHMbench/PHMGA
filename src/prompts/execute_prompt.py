"""Executor prompt contract for plan-driven DAG materialization."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .shared import render_contract_header


EXECUTE_PROMPT_INPUT_FIELDS = (
    "step_plan",
    "dag_json",
    "operator_catalog",
    "signal_context",
    "graph_path",
)
EXECUTE_PROMPT_OUTPUT_FIELDS = (
    '{"node_updates": [...], "execution_gaps": [...]}',
)
EXECUTE_PROMPT_PROHIBITIONS = (
    "invent new plan steps that are not present in step_plan",
    "change training or model internals",
    "skip unsupported steps silently",
)

EXECUTE_PROMPT_TEMPLATE = """You are an execution planner for a PHM DAG workflow.

{contract}
Task:
- Materialize `step_plan` into DAG node additions.
- Resolve missing parameters only from operator schema, signal context, or explicit reasoning.
- Only tune parameters declared in `llm_tunable_params`.
- If a step cannot be executed, emit an execution gap instead of inventing a fallback branch.

Graph path from config: {graph_path}
Signal context: {signal_context}
Current DAG: {dag_json}
Step plan: {step_plan}
Operator catalog: {operator_catalog}
"""


def render_execute_prompt(
    *,
    step_plan: Dict[str, Any],
    dag_json: Optional[Dict[str, Any]],
    operator_catalog: Iterable[Dict[str, Any]],
    signal_context: Dict[str, Any],
    graph_path: str,
) -> str:
    contract = render_contract_header(
        role="Executor",
        goal="Translate a structured step plan into legal DAG node updates.",
        input_fields=EXECUTE_PROMPT_INPUT_FIELDS,
        output_fields=EXECUTE_PROMPT_OUTPUT_FIELDS,
        prohibitions=EXECUTE_PROMPT_PROHIBITIONS,
    )
    return EXECUTE_PROMPT_TEMPLATE.format(
        contract=contract,
        graph_path=graph_path,
        signal_context=json.dumps(signal_context, ensure_ascii=False),
        dag_json=json.dumps(dag_json or {}, ensure_ascii=False),
        step_plan=json.dumps(step_plan, ensure_ascii=False),
        operator_catalog=json.dumps(list(operator_catalog), ensure_ascii=False),
    )
