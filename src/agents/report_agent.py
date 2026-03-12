"""Report agent that turns graph-dependent artifacts into final prose."""

from __future__ import annotations

from typing import Any, Dict

from src.bridge import CompiledDagManifest
from src.data import DatasetProtocol
from src.evaluation import build_final_report
from src.states import WorkflowState


def report_agent(
    state: WorkflowState,
    protocol: DatasetProtocol,
    manifest: CompiledDagManifest,
    path_artifacts: Dict[str, Any],
) -> str:
    """Build the final markdown report after path-specific execution finishes."""
    state.status = "reported"
    return build_final_report(state, protocol, manifest, path_artifacts)
