"""Reflection prompt contract for workflow-stage structural review."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable

from .shared import render_contract_header


REFLECT_PROMPT_INPUT_FIELDS = (
    "instruction",
    "stage",
    "dag_blueprint",
    "dag_quality_summary",
    "issues_summary",
    "min_depth",
    "min_width",
    "max_depth",
    "current_depth",
)
REFLECT_PROMPT_OUTPUT_FIELDS = (
    '{"decision": "finish", "reason": "...", "missing_operators": [], "shape_risks": [], "structural_warnings": []}',
    'Note: decision must be EXACTLY one of these four strings: "finish", "need_patch", "need_replan", or "halt"',
)
REFLECT_PROMPT_PROHIBITIONS = (
    "patch the DAG directly",
    "replace structural review with model metrics",
    "return plain text instead of JSON",
)

REFLECT_PROMPT_TEMPLATE = """You are an experienced PHM system architect reviewing a feature engineering DAG.

{contract}
Review guidance:
- Check structural integrity, operator legality, and planning progress.
- Use current depth and minimum depth/width as soft context, not as the only decision rule.
- Use `dag_quality_summary` to judge whether the current round is healthy enough to finish.
- If execution gaps exist, surface them explicitly in `missing_operators` or `structural_warnings`.

Instruction: {instruction}
Stage: {stage}
DAG blueprint: {dag_blueprint}
Quality summary: {dag_quality_summary}
Issues summary: {issues_summary}
Minimum depth: {min_depth}
Minimum width: {min_width}
Maximum depth: {max_depth}
Current depth: {current_depth}
"""


def render_reflect_prompt(
    *,
    instruction: str,
    stage: str,
    dag_blueprint: Dict[str, Any],
    dag_quality_summary: Dict[str, Any],
    issues_summary: str,
    min_depth: int,
    min_width: int,
    max_depth: int,
    current_depth: int,
) -> str:
    contract = render_contract_header(
        role="Reflector",
        goal="Assess whether the current DAG should finish, patch, replan, or halt.",
        input_fields=REFLECT_PROMPT_INPUT_FIELDS,
        output_fields=REFLECT_PROMPT_OUTPUT_FIELDS,
        prohibitions=REFLECT_PROMPT_PROHIBITIONS,
    )
    return REFLECT_PROMPT_TEMPLATE.format(
        contract=contract,
        instruction=instruction,
        stage=stage,
        dag_blueprint=json.dumps(dag_blueprint, ensure_ascii=False),
        dag_quality_summary=json.dumps(dag_quality_summary, ensure_ascii=False),
        issues_summary=issues_summary,
        min_depth=min_depth,
        min_width=min_width,
        max_depth=max_depth,
        current_depth=current_depth,
    )
