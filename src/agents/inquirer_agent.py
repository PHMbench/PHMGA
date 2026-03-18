"""Agent-style wrapper for downstream similarity/evidence inspection."""

from __future__ import annotations

from src.states import PHMState


def inquirer_agent(state: PHMState) -> PHMState:
    """Normalize downstream evidence after backend path execution."""

    similarity_artifacts = state.path_artifacts.get("similarity_artifacts", {})
    state.data_context["similarity_artifact_keys"] = sorted(similarity_artifacts.keys())
    state.status = "inquired"
    return state
