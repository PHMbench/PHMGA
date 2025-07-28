"""Report writer agent."""

from __future__ import annotations

import logging
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader

from ..state import PHMState

__all__ = ["write_report"]

logger = logging.getLogger(__name__)

env = Environment(loader=FileSystemLoader("templates"))


def write_report(state: PHMState) -> Dict[str, Any]:
    """Render the final Markdown report."""
    template = env.get_template("report.j2")
    report = template.render(state=state)
    return {"final_report": report}
