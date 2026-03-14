"""Shared prompt helpers and contract metadata for workflow agents."""

from __future__ import annotations

from typing import Iterable


def render_contract_header(
    *,
    role: str,
    goal: str,
    input_fields: Iterable[str],
    output_fields: Iterable[str],
    prohibitions: Iterable[str],
) -> str:
    """Render a compact contract preamble reused by all prompt templates."""

    lines = [
        f"Role: {role}",
        f"Goal: {goal}",
        "",
        "Input fields:",
    ]
    lines.extend(f"- {field}" for field in input_fields)
    lines.extend(["", "Output fields:"])
    lines.extend(f"- {field}" for field in output_fields)
    lines.extend(["", "Do not:"])
    lines.extend(f"- {item}" for item in prohibitions)
    lines.append("")
    return "\n".join(lines)
