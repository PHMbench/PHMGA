"""Offline LLM stub used by the rebuilt minimal workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class OfflineLLM:
    """Deterministic stand-in for future provider-backed prompting."""
    provider: str = "openrouter"
    mode: str = "offline_stub"

    def generate_plan(self, dataset_name: str, graph_path: str) -> List[str]:
        """Return a stable plan so tests focus on the workflow contract."""
        return [
            f"Normalize {dataset_name} metadata into the canonical protocol.",
            "Generate a compact DAG structural prior from the operator catalog.",
            f"Compile the validated DAG to the {graph_path} path artifacts.",
            "Export report-ready evidence.",
        ]

    def reflect(self, node_count: int, graph_path: str) -> str:
        """Return a compact sufficiency judgement from DAG size."""
        if node_count < 4:
            return f"DAG is too small for {graph_path}."
        return f"DAG is sufficient for {graph_path}."

    def report_summary(self, dataset_name: str, graph_path: str) -> Dict[str, Any]:
        return {"dataset": dataset_name, "graph_path": graph_path, "mode": self.mode}


def get_llm(config: Dict[str, Any]) -> OfflineLLM:
    """Resolve the currently configured LLM backend."""
    llm_cfg = dict(config.get("llm", {}))
    return OfflineLLM(
        provider=str(llm_cfg.get("provider", "openrouter")),
        mode=str(llm_cfg.get("mode", "offline_stub")),
    )
