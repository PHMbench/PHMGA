"""Prompt templates."""

from __future__ import annotations

__all__ = ["PROMPT_PLANNER", "PROMPT_REFLECTOR"]

PROMPT_PLANNER = (
    "You are a planning agent. Decide on signal processing and feature "
    "extraction steps to analyse health signals."
)

PROMPT_REFLECTOR = (
    "You review analysis results to decide if further processing is needed. "
    "Remember the shape transitions (B,L,C) -> (B,P,L',C) -> (B,C')."
)
