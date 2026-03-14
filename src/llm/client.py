"""Offline-first LLM client used by the rebuilt workflow agents.

The current repository keeps the provider contract intentionally small. The
offline stub already exposes the structured outputs that the four workflow
agents need, so unit tests can validate agent IO without requiring a remote
provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from src.states import ExecutionGap, ReflectionResult, SignalContext, StepPlan


def _leaf_node_ids(dag_json: Optional[Dict[str, Any]]) -> List[str]:
    if not dag_json:
        return []
    nodes = {node["node_id"] for node in dag_json.get("nodes", [])}
    parents = {edge["source"] for edge in dag_json.get("edges", [])}
    leaves = sorted(nodes - parents)
    return leaves


@dataclass
class OfflineLLM:
    """Deterministic stand-in for the future provider-backed client."""

    provider: str = "openrouter"
    mode: str = "offline_stub"
    model: str = "offline-stub"
    api_key_env: str = "OPENROUTER_API_KEY"

    def generate_step_plan(
        self,
        *,
        instruction: str,
        signal_context: SignalContext,
        dag_json: Optional[Dict[str, Any]],
        reflection: Iterable[str],
        graph_path: str,
        operator_catalog_summary: Iterable[Dict[str, Any]],
    ) -> StepPlan:
        """Return a deterministic NVTA-style plan for the current DAG state."""

        del instruction, reflection, operator_catalog_summary
        steps: list[dict[str, Any]] = []
        if not dag_json or not dag_json.get("nodes"):
            roots = signal_context.root_node_ids
            for root in roots:
                steps.append({"parent": root, "op_name": "normalize", "params": {"eps": 1e-6}})
            for index, root in enumerate(roots, start=1):
                steps.append({"parent": f"normalize_{index:02d}_{root}", "op_name": "fft", "params": {}})
            for index, root in enumerate(roots, start=1):
                fft_parent = f"fft_{len(roots) + index:02d}_normalize_{index:02d}_{root}"
                for feature_name in ("mean", "std", "rms"):
                    steps.append({"parent": fft_parent, "op_name": feature_name, "params": {}})
            if graph_path in {"ml", "torch"} and len(roots) > 1:
                rms_parents = [
                    f"rms_{2 * len(roots) + 3 * (index - 1) + 3:02d}_fft_{len(roots) + index:02d}_normalize_{index:02d}_{root}"
                    for index, root in enumerate(roots, start=1)
                ]
                steps.append({"parent": ",".join(rms_parents), "op_name": "concatenate", "params": {"axis": 0}})
            return StepPlan.model_validate({"plan": steps})

        leaves = _leaf_node_ids(dag_json) or signal_context.root_node_ids
        for leaf in leaves:
            if leaf.startswith("normalize_"):
                steps.append({"parent": leaf, "op_name": "fft", "params": {}})
            elif leaf.startswith("fft_"):
                steps.append({"parent": leaf, "op_name": "rms", "params": {}})
            else:
                steps.append({"parent": leaf, "op_name": "normalize", "params": {"eps": 1e-6}})
        return StepPlan.model_validate({"plan": steps})

    def resolve_missing_params(
        self,
        *,
        op_name: str,
        param_schema: Dict[str, str],
        provided_params: Dict[str, Any],
        signal_context: SignalContext,
    ) -> Dict[str, Any]:
        """Fill required params from signal metadata or conservative defaults."""

        resolved = dict(provided_params)
        for param_name in param_schema:
            if param_name in resolved:
                continue
            if param_name == "eps":
                resolved[param_name] = 1e-6
            elif param_name in {"axis"}:
                resolved[param_name] = 0
            elif param_name in {"fs", "sampling_rate"}:
                resolved[param_name] = signal_context.sampling_rate
            else:
                raise ValueError(f"Missing required parameter '{param_name}' for op '{op_name}'.")
        return resolved

    def reflect_workflow(
        self,
        *,
        instruction: str,
        stage: str,
        dag_blueprint: Dict[str, Any],
        issues_summary: str,
        min_depth: int,
        min_width: int,
        max_depth: int,
        current_depth: int,
        execution_gaps: List[ExecutionGap],
    ) -> ReflectionResult:
        """Return a structured NVTA-style review result."""

        del instruction, stage, min_width, max_depth
        missing_operators = sorted({gap.op_name for gap in execution_gaps if "Unknown" in gap.message or "unsupported" in gap.message.lower()})
        shape_risks = [gap.message for gap in execution_gaps if "shape" in gap.message.lower()]
        structural_warnings: list[str] = []
        if not dag_blueprint.get("nodes"):
            return ReflectionResult(
                decision="halt",
                reason="DAG blueprint is empty.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=["No nodes were materialized."],
            )
        if execution_gaps:
            structural_warnings.extend(gap.message for gap in execution_gaps)
            return ReflectionResult(
                decision="need_replan",
                reason=issues_summary or "Execution gaps prevent the current plan from completing cleanly.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        if current_depth < min_depth:
            return ReflectionResult(
                decision="need_patch",
                reason=f"The workflow is healthy but depth {current_depth} is below the minimum target {min_depth}.",
                structural_warnings=["Continue expanding the DAG."],
            )
        return ReflectionResult(
            decision="finish",
            reason="The DAG is structurally valid and satisfies the current planning target.",
            missing_operators=missing_operators,
            shape_risks=shape_risks,
            structural_warnings=structural_warnings,
        )

    def render_report(
        self,
        *,
        instruction: str,
        dataset_name: str,
        graph_path: str,
        compiled_manifest: Dict[str, Any],
        path_artifacts: Dict[str, Any],
        reflection_summary: Dict[str, Any],
        review_context: Dict[str, Any],
        step_plan: Dict[str, Any],
    ) -> str:
        """Generate a deterministic markdown report with path-specific sections."""

        def _keys_or_len(value: Any) -> str:
            if isinstance(value, dict):
                return ", ".join(sorted(value.keys()))
            if isinstance(value, list):
                return f"{len(value)} entries"
            return "n/a"

        lines = [
            f"# PHMGA Report: {dataset_name} / {graph_path}",
            "",
            "## Summary",
            f"- Instruction: {instruction}",
            f"- DAG hash: `{compiled_manifest['dag_hash']}`",
            f"- Path: {graph_path}",
            f"- Reflection decision: {reflection_summary.get('decision', 'unknown')}",
            "",
            "## Workflow Plan",
            f"- Planned steps: {len(step_plan.get('plan', []))}",
        ]
        if graph_path == "dag_only":
            lines.extend(
                [
                    "",
                    "## DAG Evidence",
                    f"- Node inventory: {len(path_artifacts.get('node_inventory', []))}",
                    f"- Edge inventory: {len(path_artifacts.get('edge_inventory', []))}",
                    f"- Method description: {path_artifacts.get('method_description', 'n/a')}",
                ]
            )
        elif graph_path == "ml":
            lines.extend(
                [
                    "",
                    "## ML Evidence",
                    f"- Feature specs: {len(path_artifacts.get('feature_pipeline', {}).get('feature_specs', []))}",
                    f"- Metrics keys: {', '.join(sorted(path_artifacts.get('metrics', {}).keys()))}",
                    f"- Importance keys: {', '.join(sorted(path_artifacts.get('importance', {}).keys()))}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Torch Evidence",
                    f"- Build plan keys: {_keys_or_len(path_artifacts.get('model_build_plan', {}))}",
                    f"- Training curves: {_keys_or_len(path_artifacts.get('training_curves', {}))}",
                    f"- Checkpoint keys: {_keys_or_len(path_artifacts.get('checkpoint', {}))}",
                ]
            )
        lines.extend(
            [
                "",
                "## Review Context",
                f"- Stage: {review_context.get('stage', 'FINAL_REPORT')}",
                f"- Current depth: {review_context.get('current_depth', 'n/a')}",
                f"- Issues summary: {review_context.get('issues_summary', '')}",
            ]
        )
        return "\n".join(lines) + "\n"


def get_llm(config: Dict[str, Any]) -> OfflineLLM:
    """Resolve the configured LLM client."""

    llm_cfg = dict(config.get("llm", {}))
    return OfflineLLM(
        provider=str(llm_cfg.get("provider", "openrouter")),
        mode=str(llm_cfg.get("mode", "offline_stub")),
        model=str(llm_cfg.get("model", "offline-stub")),
        api_key_env=str(llm_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
    )
