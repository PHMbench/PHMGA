"""Offline-first LLM client used by the rebuilt workflow agents.

The current repository still defaults to an offline stub, but the client
already mirrors the paper-oriented agent contracts closely enough to exercise
the main workflow:

- planner emits `StepPlan`
- executor fills operator parameters
- reflector emits `ReflectionResult`
- reporter renders graph-dependent markdown
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
        operator_catalog_summary: Iterable[Dict[str, Any]],
    ) -> StepPlan:
        """Return a deterministic NVTA-style plan for the current DAG state."""

        del instruction, reflection
        steps: list[dict[str, Any]] = []
        if not dag_json or not dag_json.get("nodes"):
            roots = signal_context.root_node_ids
            normalized: dict[str, str] = {}
            for root in roots:
                if _supports(operator_catalog_summary, "normalize"):
                    normalized[root] = _append_step(steps, root, "normalize", {"eps": 1e-6})
                else:
                    normalized[root] = root

            feature_parents: list[str] = []
            primary_root = roots[0]
            primary_signal = normalized[primary_root]

            if _supports(operator_catalog_summary, "filter"):
                filtered = _append_step(steps, primary_signal, "filter", {})
            else:
                filtered = primary_signal
            if _supports(operator_catalog_summary, "hilbert_envelope"):
                envelope = _append_step(steps, filtered, "hilbert_envelope", {})
            else:
                envelope = filtered
            if _supports(operator_catalog_summary, "kurtosis"):
                feature_parents.append(_append_step(steps, envelope, "kurtosis", {}))

            if _supports(operator_catalog_summary, "stft"):
                stft_node = _append_step(steps, primary_signal, "stft", {})
                if _supports(operator_catalog_summary, "spectral_centroid"):
                    feature_parents.append(_append_step(steps, stft_node, "spectral_centroid", {}))

            if _supports(operator_catalog_summary, "patch"):
                patch_node = _append_step(steps, primary_signal, "patch", {})
                if _supports(operator_catalog_summary, "kurtosis"):
                    feature_parents.append(_append_step(steps, patch_node, "kurtosis", {}))

            for root in roots[:2]:
                source = normalized[root]
                if _supports(operator_catalog_summary, "psd"):
                    psd_node = _append_step(steps, source, "psd", {})
                    if _supports(operator_catalog_summary, "band_power"):
                        feature_parents.append(_append_step(steps, psd_node, "band_power", {}))
                elif _supports(operator_catalog_summary, "fft"):
                    fft_node = _append_step(steps, source, "fft", {})
                    if _supports(operator_catalog_summary, "rms"):
                        feature_parents.append(_append_step(steps, fft_node, "rms", {}))

            if len(roots) > 1 and _supports(operator_catalog_summary, "cross_correlation"):
                cross_parent = f"{normalized[roots[0]]},{normalized[roots[1]]}"
                feature_parents.append(_append_step(steps, cross_parent, "cross_correlation", {}))
            elif len(roots) > 1 and _supports(operator_catalog_summary, "crest_factor"):
                feature_parents.append(_append_step(steps, normalized[roots[1]], "crest_factor", {}))

            fused_node: str | None = None
            if len(feature_parents) >= 2 and _supports(operator_catalog_summary, "concatenate"):
                fused_node = _append_step(steps, ",".join(feature_parents[:4]), "concatenate", {"axis": 0})

            if _supports(operator_catalog_summary, "threshold"):
                decision_parent = fused_node or (feature_parents[0] if feature_parents else primary_signal)
                _append_step(steps, decision_parent, "threshold", {})
            return StepPlan.model_validate({"plan": steps})

        leaves = _leaf_node_ids(dag_json) or signal_context.root_node_ids
        for leaf in leaves:
            if leaf.startswith("stft_") and _supports(operator_catalog_summary, "spectral_centroid"):
                _append_step(steps, leaf, "spectral_centroid", {})
            elif leaf.startswith("patch_") and _supports(operator_catalog_summary, "kurtosis"):
                _append_step(steps, leaf, "kurtosis", {})
            elif leaf.startswith("psd_") and _supports(operator_catalog_summary, "band_power"):
                _append_step(steps, leaf, "band_power", {})
            elif leaf.startswith("normalize_") and _supports(operator_catalog_summary, "filter"):
                _append_step(steps, leaf, "filter", {})
            elif _supports(operator_catalog_summary, "threshold") and not leaf.startswith("threshold_"):
                _append_step(steps, leaf, "threshold", {})
            elif _supports(operator_catalog_summary, "normalize"):
                _append_step(steps, leaf, "normalize", {"eps": 1e-6})
        return StepPlan.model_validate({"plan": steps})

    def resolve_missing_params(
        self,
        *,
        op_name: str,
        param_schema: Dict[str, str],
        param_defaults: Dict[str, Any],
        param_docs: Dict[str, str],
        llm_tunable_params: List[str],
        provided_params: Dict[str, Any],
        signal_context: SignalContext,
        parent_summaries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fill operator params from context, schema defaults, and simple heuristics.

        This is a deterministic stand-in for provider-backed parameter tuning.
        It only touches operator params and never training/model hyperparameters.
        """

        del param_docs
        resolved = dict(provided_params)
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
        for param_name in param_schema:
            if param_name in resolved:
                continue
            if param_name in derived:
                resolved[param_name] = derived[param_name]
                continue
            if param_name in param_defaults:
                resolved[param_name] = param_defaults[param_name]
                continue
            if param_name in llm_tunable_params:
                if param_name in derived:
                    resolved[param_name] = derived[param_name]
                    continue
            raise ValueError(f"Missing required parameter '{param_name}' for op '{op_name}'.")
        return resolved

    def reflect_workflow(
        self,
        *,
        instruction: str,
        stage: str,
        dag_blueprint: Dict[str, Any],
        dag_quality_summary: Dict[str, Any],
        issues_summary: str,
        min_depth: int,
        min_width: int,
        max_depth: int,
        current_depth: int,
        execution_gaps: List[ExecutionGap],
    ) -> ReflectionResult:
        """Return a structured NVTA-style review result."""

        del instruction, stage, min_width
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
            fatal = any(not gap.recoverable for gap in execution_gaps)
            return ReflectionResult(
                decision="need_replan" if fatal else "need_patch",
                reason=issues_summary or "Execution gaps prevent the current plan from completing cleanly.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        quality_issues = list(dag_quality_summary.get("issues", []))
        structural_warnings.extend(quality_issues)
        recommendation_hint = str(dag_quality_summary.get("recommendation_hint", ""))
        if recommendation_hint == "halt_candidate":
            return ReflectionResult(
                decision="halt",
                reason="The DAG quality summary marked the current round as halt_candidate.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        if recommendation_hint == "replan_candidate" or current_depth > max_depth:
            return ReflectionResult(
                decision="need_replan",
                reason="The DAG quality summary indicates that the current round should be replanned.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings,
            )
        depth_ok = bool(dag_quality_summary.get("depth_ok", current_depth >= min_depth))
        if recommendation_hint == "patch_candidate" or not depth_ok:
            return ReflectionResult(
                decision="need_patch",
                reason="The workflow is structurally healthy but the DAG quality summary recommends another patch round.",
                missing_operators=missing_operators,
                shape_risks=shape_risks,
                structural_warnings=structural_warnings or ["Continue expanding the DAG."],
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
        dag_quality_summary: Dict[str, Any],
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
            f"- Workflow rounds: {review_context.get('round_count', 'n/a')}",
        ]
        lines.extend(
            [
                "",
                "## DAG Quality",
                f"- Depth: {dag_quality_summary.get('current_depth', 'n/a')} / min={dag_quality_summary.get('min_depth', 'n/a')} / max={dag_quality_summary.get('max_depth', 'n/a')}",
                f"- Feature nodes: {dag_quality_summary.get('feature_node_count', 'n/a')}",
                f"- Multi nodes: {dag_quality_summary.get('multi_node_count', 'n/a')}",
                f"- Execution gaps: {dag_quality_summary.get('execution_gap_count', 'n/a')}",
                f"- NaN ratio: {dag_quality_summary.get('nan_ratio', 'n/a')}",
                f"- Zero-variance ratio: {dag_quality_summary.get('zero_variance_ratio', 'n/a')}",
                f"- Proxy probe enabled: {dag_quality_summary.get('proxy_probe_enabled', False)}",
                f"- Proxy probe macro_f1: {dag_quality_summary.get('proxy_probe_macro_f1', 'n/a')}",
            ]
        )
        decision_outputs = path_artifacts.get("decision_side_outputs", {})
        if isinstance(decision_outputs, dict) and decision_outputs:
            lines.extend(
                [
                    "",
                    "## Decision Side Outputs",
                    f"- Decision nodes: {len(decision_outputs)}",
                    f"- Keys: {', '.join(sorted(decision_outputs.keys()))}",
                ]
            )
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
                    f"- Similarity artifact keys: {', '.join(sorted(path_artifacts.get('similarity_artifacts', {}).keys()))}",
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
                    f"- Similarity artifact keys: {', '.join(sorted(path_artifacts.get('similarity_artifacts', {}).keys()))}",
                ]
            )
        lines.extend(
            [
                "",
                "## Review Context",
                f"- Stage: {review_context.get('stage', 'FINAL_REPORT')}",
                f"- Current depth: {review_context.get('current_depth', 'n/a')}",
                f"- Issues summary: {review_context.get('issues_summary', '')}",
                f"- Quality issues: {', '.join(dag_quality_summary.get('issues', []))}",
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
