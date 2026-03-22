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
1. Analyze the entire `dag_json` before proposing the next layer. Build on what already exists instead of repeating it.
2. If `dag_json` is empty, treat `signal_context.root_node_ids` as the raw multi-channel sensor roots.
3. Expand toward PHM-relevant feature diversity across multiple domains when the operator catalog allows it:
   - Time-domain analysis on raw signals for energy, impulsiveness, and distribution features.
   - Frequency-domain analysis after transforms such as `fft` or `welch` to capture spectral fault signatures.
   - Time-frequency analysis with tools such as `stft`, `wavelet_transform`, or `patch` when non-stationary behavior matters.
   - Envelope analysis with `hilbert_envelope` when rotating machinery, bearings, gears, or modulation cues are relevant.
   - Cross-channel analysis when multiple synchronous sensor roots are available and the catalog exposes legal multi-input operators.
4. Prefer logical PHM hierarchies such as raw signal -> normalize/filter -> transform -> aggregate feature extraction.
5. You may branch from any legal existing node, not just leaf nodes, when that creates a richer yet still coherent DAG.
6. Use `schema_category`, `rank_class`, `description`, and `planning_notes` to choose legal next steps that the runtime can actually execute.
7. Respect shape and rank semantics:
   - Nodes already reduced to feature vectors should not receive another layer of aggregate statistics.
   - Expand or transform operators may create new axes that later aggregate operators can legally reduce.
8. Think about the user goal. For diagnosis-oriented tasks, prefer plans that increase discriminative feature diversity instead of repeating one operator family.

Rules:
- Add a single new DAG layer only. Do not create multi-step chains within one plan item.
- Each plan item must contain `parent`, `op_name`, and `params`.
- `params` must be a JSON object, not a string.
- The only allowed top-level JSON key is `plan`. Any other top-level key is invalid.
- The plan must remain executable by the operator catalog.
- Use operator `schema_category`, `description`, and `planning_notes` to choose PHM-relevant expansions.
- Output strict JSON only.
- Do not explain the plan.
- Do not say “here is the plan”.
- Do not describe the JSON.

Instruction: {instruction}
Signal context:
{signal_context}
Current DAG:
{dag_json}
Available tools:
{tools}
Reflection:
{reflection}
Current depth: {current_depth}
Minimum depth: {min_depth}
Minimum width: {min_width}

OUTPUT FORMAT (highest priority):
Return strict JSON only:
{{"plan": [
  {{"parent": "ch1", "op_name": "hilbert_envelope", "params": {{}}}},
  {{"parent": "ch1", "op_name": "fft", "params": {{}}}},
  {{"parent": "fft_02_ch1", "op_name": "band_power", "params": {{"bands": [[0, 50], [50, 100]]}}}}
]}}
No prose before or after. No DSL. No markdown fence.
"""


SUPERVISOR_PROVING_PLAN_PROMPT_TEMPLATE = """You are building the smallest compileable PHM diagnosis DAG that can prove the end-to-end workflow works.

{contract}
Supervisor proving rules:
1. Use only the operators present in `tools`.
2. Keep the DAG small, deterministic, and easy to compile. Prefer one transform branch followed by one or more aggregate feature nodes.
3. Do not use cross-channel, multi-parent, decision, or exploratory search logic.
4. Do not depend on any operator that would require extra parameter search or repair through execution-time tuning.
5. Prefer direct single-input PHM chains such as raw signal -> `fft` or `hilbert_envelope` -> aggregate features like `rms`, `kurtosis`, or `crest_factor`.
6. Stop once the plan is sufficient to produce at least one legal aggregate feature node.
7. This proving lane exists to validate the workflow contract, not to build the richest possible DAG.

Rules:
- Each plan item must contain `parent`, `op_name`, and `params`.
- `params` must be a JSON object, not a string.
- The only allowed top-level JSON key is `plan`. Any other top-level key is invalid.
- Output strict JSON only.
- Keep the number of plan steps small.
- Do not explain the plan.
- Do not emit prose, markdown, or DSL.

Instruction: {instruction}
Signal context:
{signal_context}
Current DAG:
{dag_json}
Allowed proving tools:
{tools}
Reflection:
{reflection}
Current depth: {current_depth}
Minimum depth: {min_depth}
Minimum width: {min_width}

OUTPUT FORMAT (highest priority):
Return strict JSON only:
{{"plan": [
  {{"parent": "ch1", "op_name": "fft", "params": {{}}}},
  {{"parent": "fft_01_ch1", "op_name": "rms", "params": {{}}}}
]}}
No prose before or after. No markdown fence.
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


def _render_signal_context(signal_context: Dict[str, Any]) -> str:
    roots = ", ".join(str(item) for item in signal_context.get("root_node_ids", []) or [])
    window_shape = signal_context.get("window_shape", [])
    return "\n".join(
        [
            f"- dataset_name: {signal_context.get('dataset_name', 'unknown')}",
            f"- source_mode: {signal_context.get('source_mode', 'unknown')}",
            f"- representative_sample_id: {signal_context.get('representative_sample_id', 'unknown')}",
            f"- channel_count: {signal_context.get('channel_count', 'unknown')}",
            f"- root_node_ids: {roots or 'none'}",
            f"- sampling_rate: {signal_context.get('sampling_rate', 'unknown')}",
            f"- window_shape: {window_shape}",
        ]
    )


def _render_dag_summary(dag_json: Optional[Dict[str, Any]]) -> str:
    if not dag_json:
        return "- DAG is empty. Start from signal_context.root_node_ids."
    nodes = dag_json.get("nodes", [])
    edges = dag_json.get("edges", [])
    node_ids = ", ".join(str(node.get("node_id", "")) for node in nodes[:12] if isinstance(node, dict))
    if len(nodes) > 12:
        node_ids += ", ..."
    return "\n".join(
        [
            f"- node_count: {len(nodes)}",
            f"- edge_count: {len(edges)}",
            f"- existing_node_ids: {node_ids or 'none'}",
            "- Use existing_node_ids and operator legality to choose the next layer. Do not echo or restate DAG metadata.",
        ]
    )


def _render_tool_catalog(tools: Iterable[Dict[str, Any]]) -> str:
    rendered: list[str] = []
    for tool in _planner_tool_view(tools):
        rendered.append(
            "- {op_name} | category={schema_category} | rank={rank_class} | description={description} | notes={planning_notes}".format(
                **tool
            )
        )
    return "\n".join(rendered)


def _render_reflection_history(reflection: Iterable[str]) -> str:
    history = list(reflection)
    if not history:
        return "- none"
    return "\n".join(f"- {item}" for item in history)


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
        signal_context=_render_signal_context(signal_context),
        dag_json=_render_dag_summary(dag_json),
        tools=_render_tool_catalog(tools),
        reflection=_render_reflection_history(reflection),
        current_depth=current_depth,
        min_depth=min_depth,
        min_width=min_width,
    )


def render_supervisor_proving_plan_prompt(
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
    """Render the stricter prompt used by the lightweight proving lane."""

    contract = render_contract_header(
        role="Planner",
        goal="Produce the next structured DAG execution plan.",
        input_fields=PLAN_PROMPT_INPUT_FIELDS,
        output_fields=PLAN_PROMPT_OUTPUT_FIELDS,
        prohibitions=PLAN_PROMPT_PROHIBITIONS,
    )
    return SUPERVISOR_PROVING_PLAN_PROMPT_TEMPLATE.format(
        contract=contract,
        instruction=instruction,
        signal_context=_render_signal_context(signal_context),
        dag_json=_render_dag_summary(dag_json),
        tools=_render_tool_catalog(tools),
        reflection=_render_reflection_history(reflection),
        current_depth=current_depth,
        min_depth=min_depth,
        min_width=min_width,
    )
