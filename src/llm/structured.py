"""Pure helper logic for PHM LLM normalization, repair, and local fallbacks."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Literal, Optional

from src.states import SignalContext

from .base import LLMSchemaError


def _leaf_node_ids(dag_json: Optional[Dict[str, Any]]) -> List[str]:
    if not dag_json:
        return []
    nodes = {node["node_id"] for node in dag_json.get("nodes", [])}
    parents = {edge["source"] for edge in dag_json.get("edges", [])}
    leaves = sorted(nodes - parents)
    return leaves


def _catalog_entry(operator_catalog_summary: Iterable[Dict[str, Any]], op_name: str) -> Optional[Dict[str, Any]]:
    normalized = op_name.strip().lower()
    for item in operator_catalog_summary:
        if str(item.get("op_name", "")).strip().lower() == normalized:
            return item
    return None


def _supports(operator_catalog_summary: Iterable[Dict[str, Any]], op_name: str) -> bool:
    return _catalog_entry(operator_catalog_summary, op_name) is not None


def _planned_node_id(step_index: int, op_name: str, parent: str) -> str:
    return f"{op_name.lower()}_{step_index:02d}_{parent.replace(',', '__')}"


def _append_step(steps: list[dict[str, Any]], parent: str, op_name: str, params: dict[str, Any]) -> str:
    steps.append({"parent": parent, "op_name": op_name, "params": params})
    return _planned_node_id(len(steps), op_name, parent)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _extract_message_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMSchemaError("Provider response did not include any choices.")
    message = choices[0].get("message", {})

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        chunks.append(str(text))
                elif "text" in item:
                    chunks.append(str(item.get("text", "")))
            elif isinstance(item, str):
                chunks.append(item)
        if chunks:
            result = "".join(chunks)
            if result.strip():
                return result

    for field in ["text", "refusal", "answer", "output"]:
        alt = message.get(field)
        if isinstance(alt, str) and alt.strip():
            return alt

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        json_obj = _extract_json_object_from_text(reasoning)
        if json_obj is not None:
            return json.dumps(json_obj)
        return reasoning

    if not content or not isinstance(content, (str, list)):
        for key in message:
            if key == "role":
                continue
            val = message[key]
            if isinstance(val, str) and val.strip() and len(val) > 10:
                return val
            if isinstance(val, list) and val:
                for item in val:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            return str(text)

    raise LLMSchemaError(
        f"Provider response did not include textual message content. "
        f"Message keys: {list(message.keys())}, Content type: {type(content)}, "
        f"Content value: {repr(content)[:200]}"
    )


def _json_text_preview(text: str, limit: int = 200) -> str:
    compact = " ".join(text.strip().split())
    return compact[:limit]


def _parse_json_candidate(text: str) -> Optional[Dict[str, Any]]:
    normalized = _strip_json_fence(text)
    if not normalized:
        return None
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    direct = _parse_json_candidate(text)
    if direct is not None:
        return direct

    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in fenced_blocks:
        parsed = _parse_json_candidate(block)
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_json_object(
    text: str,
    *,
    model: str,
    structured_mode: Literal["json_mode", "text_mode"],
) -> Dict[str, Any]:
    parsed = _extract_json_object_from_text(text)
    if parsed is None:
        raise LLMSchemaError(
            "Provider response did not contain a valid JSON object. "
            f"model={model}, structured_mode={structured_mode}, text_preview={_json_text_preview(text)!r}"
        )
    return parsed


def _strip_line_prefix(line: str) -> str:
    return re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()


def _normalize_reflection_decision(raw: str) -> Optional[str]:
    normalized = raw.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "finish": "finish",
        "done": "finish",
        "complete": "finish",
        "completed": "finish",
        "finished": "finish",
        "need_patch": "need_patch",
        "patch": "need_patch",
        "continue": "need_patch",
        "continue_patch": "need_patch",
        "patch_needed": "need_patch",
        "needs_patch": "need_patch",
        "need_to_patch": "need_patch",
        "needpatch": "need_patch",
        "need_replan": "need_replan",
        "replan": "need_replan",
        "needs_replan": "need_replan",
        "replan_needed": "need_replan",
        "restart_planning": "need_replan",
        "restart": "need_replan",
        "needreplan": "need_replan",
        "halt": "halt",
        "stop": "halt",
        "abort": "halt",
        "terminate": "halt",
    }
    return aliases.get(normalized)


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return [] if len(items) == 1 and items[0].lower() in {"none", "null", "n/a"} else items
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "null", "n/a"}:
            return []
        separator = ";" if ";" in stripped and "," not in stripped else ","
        if separator in stripped:
            return [item.strip() for item in stripped.split(separator) if item.strip()]
        return [stripped]
    return [str(value).strip()]


def _parse_reflection_text_payload(text: str, *, model: str, provider: str) -> Dict[str, Any]:
    direct = _extract_json_object_from_text(text)
    if direct is not None:
        decision = _normalize_reflection_decision(str(direct.get("decision", "")))
        if decision:
            direct["decision"] = decision
        for field_name in ("missing_operators", "shape_risks", "structural_warnings"):
            direct[field_name] = _coerce_string_list(direct.get(field_name))
        return direct

    fields: Dict[str, Any] = {}
    section_map = {
        "decision": "decision",
        "reason": "reason",
        "missing operators": "missing_operators",
        "missing operator": "missing_operators",
        "missing operator(s)": "missing_operators",
        "shape risks": "shape_risks",
        "shape risk": "shape_risks",
        "shape risk(s)": "shape_risks",
        "structural warnings": "structural_warnings",
        "warnings": "structural_warnings",
        "structural issues": "structural_warnings",
    }
    active_list_field: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading = re.match(
            r"^(Decision|Reason|Missing Operators?|Missing Operator\(s\)|Shape Risks?|Shape Risk\(s\)|Structural Warnings|Warnings|Structural Issues)\s*:\s*(.*)$",
            line,
            re.I,
        )
        if heading:
            field_name = section_map[heading.group(1).strip().lower()]
            remainder = heading.group(2).strip()
            if field_name in {"missing_operators", "shape_risks", "structural_warnings"}:
                active_list_field = field_name
                existing = fields.setdefault(field_name, [])
                if remainder:
                    existing.extend(_coerce_string_list(remainder))
            else:
                active_list_field = None
                fields[field_name] = remainder
            continue
        bullet = _strip_line_prefix(line)
        if active_list_field and bullet:
            fields.setdefault(active_list_field, []).extend(_coerce_string_list(bullet))
            continue
        if "reason" in fields and active_list_field is None:
            fields["reason"] = f"{fields['reason']} {line}".strip()

    decision = _normalize_reflection_decision(str(fields.get("decision", "")))
    if not decision:
        first_lines = "\n".join([line for line in text.splitlines() if line.strip()][:3])
        decision_match = re.search(
            r"\b(finish|need_patch|need_replan|halt|patch|replan|stop|done|complete|abort|continue_patch|patch_needed|needs_patch|needs_replan|replan_needed|restart|finished)\b",
            first_lines or text,
            re.I,
        )
        decision = _normalize_reflection_decision(decision_match.group(1)) if decision_match else None
    if not decision:
        raise LLMSchemaError(
            "Provider reflection response could not be normalized into a decision. "
            f"provider={provider}, model={model}, text_preview={_json_text_preview(text)!r}"
        )

    return {
        "decision": decision,
        "reason": str(fields.get("reason", "")).strip() or "No explicit reason provided.",
        "missing_operators": _coerce_string_list(fields.get("missing_operators")),
        "shape_risks": _coerce_string_list(fields.get("shape_risks")),
        "structural_warnings": _coerce_string_list(fields.get("structural_warnings")),
    }


def _clean_plan_token(raw: str) -> str:
    return raw.strip().strip("`").strip().rstrip(",.;")


def _parse_plan_params(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()
    if not cleaned or cleaned.lower() in {"{}", "none", "null", "n/a"}:
        return {}
    if cleaned[:1] in {"'", '"'} and cleaned[-1:] == cleaned[:1]:
        cleaned = cleaned[1:-1]
    parsed = _parse_json_candidate(cleaned)
    return parsed if parsed is not None else {}


def _extract_block_step(block_text: str) -> Optional[Dict[str, Any]]:
    parent_match = re.search(
        r"(?:^|\n)\s*(?:parent|input|source)\s*(?:=|:)\s*(?P<value>[A-Za-z0-9_,.-]+)",
        block_text,
        re.I,
    )
    op_match = re.search(
        r"(?:^|\n)\s*(?:op|operator|operation|op_name)\s*(?:=|:)\s*(?P<value>[A-Za-z0-9_.-]+)",
        block_text,
        re.I,
    )
    params_match = re.search(
        r"(?:^|\n)\s*(?:params?|parameters?)\s*(?:=|:)\s*(?P<value>\{.*?\}|\".*?\"|'.*?')",
        block_text,
        re.I | re.S,
    )
    if not (parent_match and op_match):
        return None
    return {
        "parent": _clean_plan_token(parent_match.group("value")),
        "op_name": _clean_plan_token(op_match.group("value")),
        "params": _parse_plan_params(params_match.group("value")) if params_match else {},
    }


def _parse_plan_text_payload(
    text: str,
    *,
    model: str,
    provider: str,
    allow_text_fallback: bool = False,
) -> Dict[str, Any]:
    direct = _extract_json_object_from_text(text)
    if direct is not None:
        if "plan" in direct:
            plan_items = direct.get("plan")
            if isinstance(plan_items, list):
                sanitized_steps: List[Dict[str, Any]] = []
                for item in plan_items:
                    if not isinstance(item, dict):
                        continue
                    params = item.get("params", {})
                    if params == "" or params is None:
                        params = {}
                    if not isinstance(params, dict):
                        raise LLMSchemaError(
                            "Provider planner response contained a non-object params field. "
                            f"provider={provider}, model={model}, text_preview={_json_text_preview(text)!r}"
                        )
                    sanitized_steps.append(
                        {
                            "parent": str(item.get("parent", "")).strip(),
                            "op_name": str(item.get("op_name", "")).strip(),
                            "params": params,
                        }
                    )
                return {"plan": sanitized_steps}
        if {"parent", "op_name"}.issubset(direct.keys()):
            params = direct.get("params", {})
            if params == "" or params is None:
                params = {}
            if not isinstance(params, dict):
                raise LLMSchemaError(
                    "Provider planner response contained a non-object params field. "
                    f"provider={provider}, model={model}, text_preview={_json_text_preview(text)!r}"
                )
            return {
                "plan": [
                    {
                        "parent": str(direct.get("parent", "")).strip(),
                        "op_name": str(direct.get("op_name", "")).strip(),
                        "params": params,
                    }
                ]
            }

    if not allow_text_fallback:
        raise LLMSchemaError(
            "Provider planner response did not contain strict StepPlan JSON. "
            f"provider={provider}, model={model}, text_preview={_json_text_preview(text)!r}"
        )

    steps: List[Dict[str, Any]] = []
    text = re.sub(r"```(?:text|txt)?\s*(.*?)```", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    line_pattern = re.compile(
        r"^\s*(?:[-*]|\d+[.)])?\s*parent\s*(?:=|:)\s*(?P<parent>.+?)\s+"
        r"(?:op|operator|op_name)\s*(?:=|:)\s*(?P<op>[^\s]+)\s+"
        r"params\s*(?:=|:)\s*(?P<params>.+?)\s*$",
        re.I,
    )
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = line_pattern.match(line)
        if not match:
            continue
        raw_params = match.group("params").strip()
        if raw_params[:1] in {"'", '"'} and raw_params[-1:] == raw_params[:1]:
            raw_params = raw_params[1:-1]
        params = _parse_json_candidate(raw_params)
        if params is None:
            continue
        steps.append(
            {
                "parent": match.group("parent").strip(),
                "op_name": match.group("op").strip(),
                "params": params,
            }
        )
    if steps:
        return {"plan": steps}

    compact_patterns = [
        re.compile(
            r"^\s*(?:[-*]|\d+[.)])\s*(?P<op>[A-Za-z_][\w.]*)\s+(?:to|on|from|for)\s+(?P<parent>[A-Za-z0-9_,.-]+)"
            r"(?:\s+(?:with\s+)?params?\s*(?:=|:)?\s*(?P<params>\{.*\}|\".*\"|'.*'))?\s*$",
            re.I,
        ),
        re.compile(
            r"^\s*(?:[-*]|\d+[.)])\s*(?P<parent>[A-Za-z0-9_,.-]+)\s*->\s*(?P<op>[A-Za-z_][\w.]*)"
            r"(?:\s+(?:with\s+)?params?\s*(?:=|:)?\s*(?P<params>\{.*\}|\".*\"|'.*'))?\s*$",
            re.I,
        ),
        re.compile(
            r"^\s*(?:[-*]|\d+[.)])\s*(?:apply|use|run|compute|extract)\s+(?P<op>[A-Za-z_][\w.]*)\s+"
            r"(?:to|on|from)\s+(?P<parent>[A-Za-z0-9_,.-]+)"
            r"(?:\s+(?:with\s+)?params?\s*(?:=|:)?\s*(?P<params>\{.*\}|\".*\"|'.*'))?\s*$",
            re.I,
        ),
    ]
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for pattern in compact_patterns:
            match = pattern.match(line)
            if not match:
                continue
            steps.append(
                {
                    "parent": _clean_plan_token(match.group("parent")),
                    "op_name": _clean_plan_token(match.group("op")),
                    "params": _parse_plan_params(match.group("params") or ""),
                }
            )
            break
    if steps:
        return {"plan": steps}

    block_lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if block_lines:
        blocks: List[str] = []
        current: List[str] = []
        for line in block_lines:
            if re.match(r"^\s*(?:step\s+\d+[:.)]?|\d+[.)])\s*", line, re.I):
                if current:
                    blocks.append("\n".join(current))
                    current = []
                line = re.sub(r"^\s*(?:step\s+\d+[:.)]?|\d+[.)])\s*", "", line, flags=re.I)
            current.append(line)
        if current:
            blocks.append("\n".join(current))
        for block in blocks:
            parsed = _extract_block_step(block)
            if parsed is not None:
                steps.append(parsed)
    if steps:
        return {"plan": steps}

    raise LLMSchemaError(
        "Provider planner response could not be normalized into StepPlan. "
        f"provider={provider}, model={model}, text_preview={_json_text_preview(text)!r}"
    )


def _param_json_schema(param_type: str) -> Dict[str, Any]:
    normalized = param_type.strip().lower()
    if any(token in normalized for token in ("float", "double", "number")):
        return {"type": "number"}
    if any(token in normalized for token in ("int", "integer")):
        return {"type": "integer"}
    if any(token in normalized for token in ("bool", "boolean")):
        return {"type": "boolean"}
    if any(token in normalized for token in ("dict", "object", "map", "json")):
        return {"type": "object", "properties": {}, "additionalProperties": False}
    if any(token in normalized for token in ("list", "array", "sequence", "tuple")):
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


def _codex_reflection_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["finish", "need_patch", "need_replan", "halt"]},
            "reason": {"type": "string"},
            "missing_operators": {"type": "array", "items": {"type": "string"}},
            "shape_risks": {"type": "array", "items": {"type": "string"}},
            "structural_warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["decision", "reason", "missing_operators", "shape_risks", "structural_warnings"],
        "additionalProperties": False,
    }


def _derived_param_candidates(signal_context: SignalContext, parent_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    window_length = int(signal_context.window_shape[-1]) if signal_context.window_shape else 128
    patch_length = max(16, min(128, max(window_length // 8, 16)))
    derived: Dict[str, Any] = {
        "fs": signal_context.sampling_rate,
        "sampling_rate": signal_context.sampling_rate,
        "nperseg": max(16, min(128, window_length // 4 or window_length)),
        "noverlap": max(8, min(64, window_length // 8 or 8)),
        "patch_length": patch_length,
        "stride": max(8, patch_length // 2),
        "band_low_hz": max(5.0, signal_context.sampling_rate * 0.02),
        "band_high_hz": max(20.0, signal_context.sampling_rate * 0.2),
        "low_cut_hz": max(5.0, signal_context.sampling_rate * 0.02),
        "high_cut_hz": max(20.0, signal_context.sampling_rate * 0.2),
        "threshold": 0.5,
        "mode": "bandpass",
        "order": 4,
        "axis": 0,
    }
    if parent_summaries and any("shape" in item for item in parent_summaries):
        derived["parent_count"] = len(parent_summaries)
    return derived


def _deterministic_param_resolution(
    *,
    op_name: str,
    param_schema: Dict[str, str],
    param_defaults: Dict[str, Any],
    llm_tunable_params: List[str],
    provided_params: Dict[str, Any],
    signal_context: SignalContext,
    parent_summaries: List[Dict[str, Any]],
    provider_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved = dict(provided_params)
    derived = _derived_param_candidates(signal_context, parent_summaries)

    for param_name in param_schema:
        if param_name in resolved:
            continue
        if param_name in derived:
            resolved[param_name] = derived[param_name]
            continue
        if param_name in param_defaults:
            resolved[param_name] = param_defaults[param_name]
            continue

    if provider_params:
        illegal = sorted(set(provider_params) - set(llm_tunable_params))
        if illegal:
            raise LLMSchemaError(
                f"Provider attempted to set non-tunable params for '{op_name}': {illegal}"
            )
        for param_name, value in provider_params.items():
            if param_name not in resolved:
                resolved[param_name] = value

    missing = [param_name for param_name in param_schema if param_name not in resolved]
    if missing:
        raise ValueError(f"Missing required parameter(s) {missing} for op '{op_name}'.")
    return resolved


def _local_param_resolution(
    *,
    param_schema: Dict[str, str],
    param_defaults: Dict[str, Any],
    provided_params: Dict[str, Any],
    signal_context: SignalContext,
    parent_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    resolved = dict(provided_params)
    derived = _derived_param_candidates(signal_context, parent_summaries)
    for param_name in param_schema:
        if param_name in resolved:
            continue
        if param_name in derived:
            resolved[param_name] = derived[param_name]
            continue
        if param_name in param_defaults:
            resolved[param_name] = param_defaults[param_name]
    return resolved


def _repair_prompt(*, task: Literal["plan", "reflect"], original_prompt: str, raw_response: str) -> str:
    if task == "plan":
        return (
            "Normalize the following planner response into a machine-readable step plan.\n"
            "Return either a strict JSON object with a top-level `plan` list, or the exact DSL lines:\n"
            "`- parent=<node_id_or_csv> op=<operator_name> params=<json_object>`\n\n"
            "Use only the minimum fields required for StepPlan: `parent`, `op_name`, `params`.\n"
            "`params` must be a JSON object, not a string. If parameters are unspecified, use `{}`.\n"
            "If the original response is prose, convert it into the DSL instead of explaining it.\n\n"
            f"Original planner contract:\n{original_prompt}\n\n"
            f"Raw model response:\n{raw_response}\n"
        )
    return (
        "Normalize the following reflection response into a machine-readable structural review.\n"
        "Return either a strict JSON object with fields `decision`, `reason`, `missing_operators`, "
        "`shape_risks`, `structural_warnings`, or the fallback headings:\n"
        "Decision:\nReason:\nMissing Operators:\nShape Risks:\nStructural Warnings:\n\n"
        "Allowed decisions are: finish, need_patch, need_replan, halt.\n\n"
        f"Original reflection contract:\n{original_prompt}\n\n"
        f"Raw model response:\n{raw_response}\n"
    )


__all__ = [
    "_append_step",
    "_catalog_entry",
    "_codex_reflection_schema",
    "_coerce_string_list",
    "_derived_param_candidates",
    "_deterministic_param_resolution",
    "_extract_json_object_from_text",
    "_extract_message_text",
    "_json_text_preview",
    "_leaf_node_ids",
    "_local_param_resolution",
    "_normalize_reflection_decision",
    "_param_json_schema",
    "_parse_json_candidate",
    "_parse_json_object",
    "_parse_plan_text_payload",
    "_parse_reflection_text_payload",
    "_planned_node_id",
    "_repair_prompt",
    "_strip_json_fence",
    "_strip_line_prefix",
    "_supports",
]
