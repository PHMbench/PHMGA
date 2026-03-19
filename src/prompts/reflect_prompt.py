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
    'Fallback text format: `Decision: <finish|need_patch|need_replan|halt>` followed by `Reason:` and optional bullet sections',
)
REFLECT_PROMPT_PROHIBITIONS = (
    "patch the DAG directly",
    "replace structural review with model metrics",
    "return unconstrained prose without a clear decision and supporting fields",
    "discuss the allowed decision values instead of choosing one",
)

REFLECT_PROMPT_TEMPLATE = """You are an experienced PHM system architect reviewing a feature engineering DAG.

{contract}
Review guidance:
- Check structural integrity, operator legality, and planning progress.
- Use current depth and minimum depth/width as soft context, not as the only decision rule.
- Use `dag_quality_summary` to judge whether the current round is healthy enough to finish.
- Treat `dag_quality_summary.dataset_level` as stronger evidence than representative preview-only signals when it exists.
- If execution gaps exist, surface them explicitly in `missing_operators` or `structural_warnings`.
- Evaluate operator diversity. A healthy PHM DAG should not collapse into repetitive aggregate-only branches.
- Evaluate hierarchy. Prefer workflows like signal -> transform -> feature extraction, and flag obviously misplaced operators.
- Evaluate redundancy and symmetry. Recent duplicate branches or asymmetric multi-channel handling are valid reasons for `need_patch` or `need_replan`.
- Keep the `reason` actionable. It should help the planner decide what kind of next step is required.
- If the DAG is legal and usable, choose `finish` instead of describing alternatives.
- Do not discuss the allowed values. Choose one decision.

Instruction: {instruction}
Stage: {stage}
DAG blueprint: {dag_blueprint}
Quality summary: {dag_quality_summary}
Issues summary: {issues_summary}
Minimum depth: {min_depth}
Minimum width: {min_width}
Maximum depth: {max_depth}
Current depth: {current_depth}

OUTPUT FORMAT (highest priority):
1. Preferred output is strict JSON:
{{"decision": "finish", "reason": "brief actionable reason", "missing_operators": [], "shape_risks": [], "structural_warnings": []}}
2. If strict JSON is not possible, output structured text using exactly these headings:
Decision: <finish|need_patch|need_replan|halt>
Reason: <one actionable sentence>
Missing Operators:
- <operator name>
Shape Risks:
- <risk>
Structural Warnings:
- <warning>
3. Example fallback:
Decision: need_patch
Reason: add a transform branch before feature aggregation
Missing Operators:
- stft
Shape Risks:
Structural Warnings:
4. Output only JSON or the heading-based fallback. No prose before or after.
5. Allowed decisions: finish, need_patch, need_replan, halt
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
