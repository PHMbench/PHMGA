"""Planner-trace helpers for provider debugging and Stage B incident review."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils import ensure_dir, write_json, write_text


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trace_dir(trace_context: Optional[Dict[str, Any]]) -> Optional[Path]:
    if not trace_context:
        return None
    output_dir = str(trace_context.get("output_dir", "")).strip()
    if not output_dir:
        return None
    return ensure_dir(output_dir)


def write_planner_text_artifact(
    trace_context: Optional[Dict[str, Any]],
    *,
    filename: str,
    content: str,
) -> Optional[str]:
    trace_dir = _trace_dir(trace_context)
    if trace_dir is None:
        return None
    write_text(content, trace_dir / filename)
    return filename


def append_planner_trace_event(
    trace_context: Optional[Dict[str, Any]],
    *,
    filename: str,
    provider: str,
    model: str,
    event: Dict[str, Any],
) -> Optional[str]:
    trace_dir = _trace_dir(trace_context)
    if trace_dir is None:
        return None
    trace_path = trace_dir / filename
    if trace_path.exists():
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "provider": provider,
            "model": model,
            "events": [],
        }
    payload["provider"] = provider
    payload["model"] = model
    payload.setdefault("events", []).append({"timestamp": _now_iso(), **event})
    write_json(payload, trace_path)
    return filename


__all__ = ["append_planner_trace_event", "write_planner_text_artifact"]
