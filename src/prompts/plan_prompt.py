"""Planner prompt contract derived from NVTA-style DAG expansion."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from .shared import render_contract_header


PLAN_PROMPT_INPUT_FIELDS = (
    "instruction",
    "signal_context",
    "dag_json",
    "tools",
    "reflection",
    "current_depth",
    "min_depth",
    "min_width",
)
PLAN_PROMPT_OUTPUT_FIELDS = ('{"plan": [{"parent": "...", "op_name": "...", "params": {...}}]}',)
PLAN_PROMPT_PROHIBITIONS = (
    "invent operators that are not present in tools",
    "modify model or training parameters",
    "emit free-form prose instead of JSON",
)

PLAN_PROMPT_TEMPLATE = """You are a world-class AI strategist specializing in signal processing for Prognostics and Health Management (PHM).

{contract}
Strategic guidance:
1. Analyze the existing DAG before proposing the next steps.
2. If `dag_json` is empty, treat `signal_context.root_node_ids` as the multi-channel raw signal roots.
3. Prefer PHM-relevant workflows such as time-domain -> frequency-domain -> feature extraction.
4. Use any existing node as a parent when that grows the DAG logically.
5. Respect signal shapes and rank behavior. Do not apply aggregate statistics to nodes that are already reduced features.
6. Use `schema_category`, `rank_class`, `input_spec`, and `output_spec` to choose legal next steps.

Rules:
- Return valid JSON only.
- Each plan item must contain `parent`, `op_name`, and `params`.
- The plan must remain executable by the operator catalog.
- Use operator `schema_category`, `description`, and `planning_notes` to choose PHM-relevant expansions.

Instruction: {instruction}
Signal context: {signal_context}
Current DAG: {dag_json}
Available tools: {tools}
Reflection: {reflection}
Current depth: {current_depth}
Minimum depth: {min_depth}
Minimum width: {min_width}
"""


def render_plan_prompt(
    *,
    instruction: str,
    signal_context: Dict[str, Any],
    dag_json: Optional[Dict[str, Any]],
    tools: Iterable[Dict[str, Any]],
    reflection: Iterable[str],
    current_depth: int,
    min_depth: int,
    min_width: int,
) -> str:
    """Render the planner prompt with explicit IO and rule sections."""

    contract = render_contract_header(
        role="Planner",
        goal="Produce the next structured DAG execution plan.",
        input_fields=PLAN_PROMPT_INPUT_FIELDS,
        output_fields=PLAN_PROMPT_OUTPUT_FIELDS,
        prohibitions=PLAN_PROMPT_PROHIBITIONS,
    )
    return PLAN_PROMPT_TEMPLATE.format(
        contract=contract,
        instruction=instruction,
        signal_context=json.dumps(signal_context, ensure_ascii=False),
        dag_json=json.dumps(dag_json or {}, ensure_ascii=False),
        tools=json.dumps(list(tools), ensure_ascii=False),
        reflection=json.dumps(list(reflection), ensure_ascii=False),
        current_depth=current_depth,
        min_depth=min_depth,
        min_width=min_width,
    )
