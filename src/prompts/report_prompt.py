"""Report prompt contract for graph-dependent PHM reporting."""

from __future__ import annotations

import json
from typing import Any, Dict

from .shared import render_contract_header


REPORT_PROMPT_INPUT_FIELDS = (
    "instruction",
    "graph_path",
    "compiled_manifest",
    "path_artifacts",
    "reflection_summary",
    "dag_quality_summary",
    "review_context",
)
REPORT_PROMPT_OUTPUT_FIELDS = ("markdown report",)
REPORT_PROMPT_PROHIBITIONS = (
    "claim a full research-grade torch trainer when artifacts come from the minimal torch tensor runtime",
    "reference artifacts that do not exist for the current graph path",
    "emit non-markdown output",
)

REPORT_PROMPT_TEMPLATE = """You are a PHM research report engineer.

{contract}
Write a graph-dependent report using artifacts as the primary evidence source and review context as auxiliary context.
Return markdown only. No preface. No code fences unless showing code.
Use this section scaffold:
# Summary
# DAG
# Path Artifacts
# Dataset-Level Diagnosis Evidence
# Diagnostics
# Limitations

Writing guidance:
- Start with a concise process overview that explains how the DAG supports the diagnosis objective.
- Use the compiled manifest and path artifacts as the primary factual source of truth.
- Use reflection and dag-quality outputs as review context, not as substitutes for missing evidence.
- Highlight the strongest available diagnosis evidence and the main operational caveats.
- If an artifact is missing, say it is missing. Do not infer it.
- Keep the report consistent with the actual graph path. Do not claim a richer backend or training stack than the artifacts support.
- When discussing limitations, convert them into concrete PHM-oriented next-step recommendations when possible.

Instruction: {instruction}
Graph path: {graph_path}
Compiled manifest: {compiled_manifest}
Path artifacts: {path_artifacts}
Reflection summary: {reflection_summary}
Dag quality summary: {dag_quality_summary}
Review context: {review_context}
"""


def render_report_prompt(
    *,
    instruction: str,
    graph_path: str,
    compiled_manifest: Dict[str, Any],
    path_artifacts: Dict[str, Any],
    reflection_summary: Dict[str, Any],
    dag_quality_summary: Dict[str, Any],
    review_context: Dict[str, Any],
) -> str:
    contract = render_contract_header(
        role="Reporter",
        goal="Generate the final graph-dependent PHM report.",
        input_fields=REPORT_PROMPT_INPUT_FIELDS,
        output_fields=REPORT_PROMPT_OUTPUT_FIELDS,
        prohibitions=REPORT_PROMPT_PROHIBITIONS,
    )
    return REPORT_PROMPT_TEMPLATE.format(
        contract=contract,
        instruction=instruction,
        graph_path=graph_path,
        compiled_manifest=json.dumps(compiled_manifest, ensure_ascii=False),
        path_artifacts=json.dumps(path_artifacts, ensure_ascii=False),
        reflection_summary=json.dumps(reflection_summary, ensure_ascii=False),
        dag_quality_summary=json.dumps(dag_quality_summary, ensure_ascii=False),
        review_context=json.dumps(review_context, ensure_ascii=False),
    )
