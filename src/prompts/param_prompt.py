"""Prompt contract for provider-backed operator parameter completion."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from .shared import render_contract_header


PARAM_PROMPT_INPUT_FIELDS = (
    "op_name",
    "requested_params",
    "param_schema",
    "param_defaults",
    "param_docs",
    "llm_tunable_params",
    "provided_params",
    "signal_context",
    "parent_summaries",
)
PARAM_PROMPT_OUTPUT_FIELDS = ('{"param_name": value, "...": value}',)
PARAM_PROMPT_PROHIBITIONS = (
    "invent keys outside requested_params",
    "change graph structure or model hyperparameters",
    "return prose or markdown instead of a JSON object",
)

PARAM_PROMPT_TEMPLATE = """You are a PHM operator parameter specialist.

{contract}
Task:
- Only fill the operator params listed in `requested_params`.
- Use `signal_context`, `parent_summaries`, and operator docs to infer plausible PHM-oriented values.
- Respect the declared `param_schema`.
- Return a JSON object containing only the requested keys that you can justify.

Operator name: {op_name}
Requested params: {requested_params}
Parameter schema: {param_schema}
Parameter defaults: {param_defaults}
Parameter docs: {param_docs}
LLM-tunable params: {llm_tunable_params}
Provided params: {provided_params}
Signal context: {signal_context}
Parent summaries: {parent_summaries}
"""


def render_param_resolution_prompt(
    *,
    op_name: str,
    requested_params: List[str],
    param_schema: Dict[str, str],
    param_defaults: Dict[str, Any],
    param_docs: Dict[str, str],
    llm_tunable_params: Iterable[str],
    provided_params: Dict[str, Any],
    signal_context: Dict[str, Any],
    parent_summaries: List[Dict[str, Any]],
) -> str:
    """Render the narrow provider prompt used for unresolved operator params."""

    contract = render_contract_header(
        role="Parameter Resolver",
        goal="Fill only the unresolved, LLM-tunable operator parameters.",
        input_fields=PARAM_PROMPT_INPUT_FIELDS,
        output_fields=PARAM_PROMPT_OUTPUT_FIELDS,
        prohibitions=PARAM_PROMPT_PROHIBITIONS,
    )
    return PARAM_PROMPT_TEMPLATE.format(
        contract=contract,
        op_name=op_name,
        requested_params=json.dumps(requested_params, ensure_ascii=False),
        param_schema=json.dumps(param_schema, ensure_ascii=False),
        param_defaults=json.dumps(param_defaults, ensure_ascii=False),
        param_docs=json.dumps(param_docs, ensure_ascii=False),
        llm_tunable_params=json.dumps(list(llm_tunable_params), ensure_ascii=False),
        provided_params=json.dumps(provided_params, ensure_ascii=False),
        signal_context=json.dumps(signal_context, ensure_ascii=False),
        parent_summaries=json.dumps(parent_summaries, ensure_ascii=False),
    )
