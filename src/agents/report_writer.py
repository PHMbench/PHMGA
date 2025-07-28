from __future__ import annotations

import logging
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader

from ..state import PHMState

__all__ = ["write_report"]


def write_report(state: PHMState) -> Dict[str, Any]:
    """Render the final report."""
    logging.info("Writing report")
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.j2")
    report = template.render(state=state)
    return {"final_report": report}
