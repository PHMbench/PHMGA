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
PLAN_PROMPT_OUTPUT_FIELDS = (
    '{"plan": [{"parent": "...", "op_name": "...", "params": {...}}]}',
)
PLAN_PROMPT_PROHIBITIONS = (
    "invent operators that are not present in tools",
    "modify model or training parameters",
    "emit any DSL or free-form text instead of strict JSON",
    "emit unconstrained prose that cannot be parsed as StepPlan JSON",
    "describe the output format instead of producing the JSON itself",
)

PLAN_PROMPT_TEMPLATE = """You are a world-class AI strategist specializing in signal processing for Prognostics and Health Management (PHM).

{contract}
Strategic guidance:
1. Analyze the existing DAG before proposing the next steps.
2. If `dag_json` is empty, treat `signal_context.root_node_ids` as the multi-channel raw signal roots.
3. Build PHM-relevant feature diversity across time-domain, frequency-domain, time-frequency, envelope, and cross-channel branches when that improves the DAG.
4. Prefer workflows such as raw signal -> normalize/filter -> spectral or time-frequency transform -> aggregate features.
5. If the task mentions rotating machinery, bearings, gears, transients, or modulation, consider `hilbert_envelope`, `stft`, `patch`, and cross-channel analysis when available.
6. Use any existing node as a parent when that grows the DAG logically.
7. Respect signal shapes and rank behavior. Do not apply aggregate statistics to nodes that are already reduced features.
8. Use `schema_category`, `rank_class`, `description`, and `planning_notes` to choose legal next steps.

Rules:
- Each plan item must contain `parent`, `op_name`, and `params`.
- `params` must be a JSON object, not a string.
- The plan must remain executable by the operator catalog.
- Use operator `schema_category`, `description`, and `planning_notes` to choose PHM-relevant expansions.
- Output strict JSON only.
- Do not explain the plan.
- Do not say “here is the plan”.
- Do not describe the JSON.

Instruction: {instruction}
Signal context: {signal_context}
Current DAG: {dag_json}
Available tools: {tools}
Reflection: {reflection}
Current depth: {current_depth}
Minimum depth: {min_depth}
Minimum width: {min_width}

OUTPUT FORMAT (highest priority):
Return strict JSON only:
{{"plan": [{{"parent": "ch1", "op_name": "normalize", "params": {{"eps": 1e-6}}}}]}}
No prose before or after. No DSL. No markdown fence.
"""


def _planner_tool_view(tools: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    compact: list[Dict[str, Any]] = []
    for tool in tools:
        compact.append(
            {
                "op_name": tool.get("op_name"),
                "schema_category": tool.get("schema_category"),
                "rank_class": tool.get("rank_class"),
                "description": str(tool.get("description", "")).strip(),
                "planning_notes": str(tool.get("planning_notes", "")).strip(),
            }
        )
    return compact


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
        tools=json.dumps(_planner_tool_view(tools), ensure_ascii=False),
        reflection=json.dumps(list(reflection), ensure_ascii=False),
        current_depth=current_depth,
        min_depth=min_depth,
        min_width=min_width,
    )
