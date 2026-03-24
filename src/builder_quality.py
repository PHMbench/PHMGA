from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List

from src.states.phm_states import PHMState
from src.tools.signal_processing_schemas import AggregateOp, MultiVariableOp, get_operator
from src.utils import get_dag_depth


PHASE_RAW = "raw_phase"
PHASE_FEATURE = "feature_phase"
PHASE_COMBINE = "combine_phase"

FAMILY_PREPROCESS = "preprocess"
FAMILY_SPECTRAL = "spectral_transform"
FAMILY_FEATURE = "feature_stat"
FAMILY_COMBINE = "cross_or_combine"
FAMILY_UNKNOWN = "unknown"

WEAK_REDUCERS = {"mean", "std", "min", "max", "var", "abs_mean"}

RAW_SHORTLIST = ["psd", "stft", "hilbert_envelope", "fft", "normalize"]
FEATURE_SHORTLIST = ["band_power", "spectral_centroid", "rms", "kurtosis", "crest_factor"]
COMBINE_SHORTLIST = ["concatenate", "arithmetic", "cross_correlation", "coherence"]

RAW_FALLBACK_TRANSFORMS = ["psd", "hilbert_envelope", "stft", "fft"]
FEATURE_FALLBACK_OPS = ["band_power", "kurtosis", "spectral_centroid", "rms"]
COMBINE_FALLBACK_OPS = ["concatenate", "arithmetic"]

HARD_ERROR_MARKERS = (
    "not found",
    "unknown operator",
    "invalid parent",
    "single-variable but received multiple parents",
    "cycle detected",
)


def get_op_family(op_name: str) -> str:
    normalized = str(op_name or "").strip()
    if not normalized:
        return FAMILY_UNKNOWN
    if normalized == "normalize":
        return FAMILY_PREPROCESS
    try:
        op_cls = get_operator(normalized)
    except KeyError:
        return FAMILY_UNKNOWN
    if issubclass(op_cls, AggregateOp):
        return FAMILY_FEATURE
    if issubclass(op_cls, MultiVariableOp):
        return FAMILY_COMBINE
    if normalized in {"fft", "psd", "stft", "hilbert_envelope"}:
        return FAMILY_SPECTRAL
    return FAMILY_PREPROCESS


def get_node_op_name(node: Any) -> str:
    return str(node.meta.get("tool") or node.meta.get("method") or getattr(node, "method", "") or "").strip()


def is_feature_leaf(state: PHMState, node_id: str) -> bool:
    node = state.dag_state.nodes[node_id]
    if node.stage == "input":
        return False
    op_name = get_node_op_name(node)
    family = get_op_family(op_name)
    if family == FAMILY_FEATURE:
        return True
    if family != FAMILY_COMBINE:
        return False
    parent_ids = node.parents if isinstance(node.parents, list) else [node.parents]
    return all(parent_id in state.dag_state.nodes and is_feature_leaf(state, parent_id) for parent_id in parent_ids if parent_id)


def infer_builder_phase(state: PHMState) -> str:
    leaves = [leaf for leaf in state.dag_state.leaves if leaf in state.dag_state.nodes]
    if not leaves or all(state.dag_state.nodes[leaf].stage == "input" for leaf in leaves):
        return PHASE_RAW

    transform_leaves = [leaf for leaf in leaves if state.dag_state.nodes[leaf].stage != "input" and not is_feature_leaf(state, leaf)]
    if transform_leaves:
        return PHASE_FEATURE

    if get_dag_depth(state.dag_state) < state.min_depth:
        return PHASE_COMBINE
    return PHASE_FEATURE


def shortlist_for_phase(phase: str) -> list[str]:
    if phase == PHASE_RAW:
        return list(RAW_SHORTLIST)
    if phase == PHASE_FEATURE:
        return list(FEATURE_SHORTLIST)
    if phase == PHASE_COMBINE:
        return list(COMBINE_SHORTLIST)
    return list(RAW_SHORTLIST)


def compact_tool_descriptions_for_phase(phase: str) -> str:
    lines: list[str] = []
    for op_name in shortlist_for_phase(phase):
        try:
            op_cls = get_operator(op_name)
        except KeyError:
            continue
        schema = op_cls.model_json_schema()
        properties = schema.get("properties", {})
        required = [
            name
            for name in schema.get("required", [])
            if name not in {"parent", "op_name", "description", "input_spec", "output_spec"}
        ]
        optional = [
            name
            for name in properties
            if name not in {"parent", "op_name", "description", "input_spec", "output_spec"} and name not in required
        ]
        lines.append(
            f"- {schema.get('title', op_name)}"
            f" | family={get_op_family(op_name)}"
            f" | required_params={required or []}"
            f" | optional_params={optional or []}"
        )
    return "\n".join(lines)


def _plan_parent_ids(step: Dict[str, Any]) -> list[str]:
    parent_value = step.get("parent", "")
    if isinstance(parent_value, list):
        return [str(item).strip() for item in parent_value if str(item).strip()]
    return [segment.strip() for segment in str(parent_value).split(",") if segment.strip()]


def validate_plan_steps(state: PHMState, steps: List[Dict[str, Any]], *, phase: str | None = None) -> tuple[bool, str]:
    phase = phase or infer_builder_phase(state)
    if not steps:
        return False, "planner produced an empty plan"

    families = {get_op_family(str(step.get("op_name", "")).strip()) for step in steps}
    op_names = {str(step.get("op_name", "")).strip() for step in steps if str(step.get("op_name", "")).strip()}
    parent_ids = {parent_id for step in steps for parent_id in _plan_parent_ids(step)}
    if any(parent_id not in state.dag_state.nodes for parent_id in parent_ids):
        return False, "plan references unknown parent nodes"
    if len(parent_ids) < max(1, int(state.min_width)):
        return False, "plan does not cover enough processing branches"
    if op_names == {"mean"}:
        return False, "plan is mean-only"
    if len(op_names) == 1 and op_names.issubset(WEAK_REDUCERS):
        return False, "plan relies on a single weak reducer"

    if phase == PHASE_RAW:
        if FAMILY_SPECTRAL not in families:
            return False, "raw-phase plan must contain at least one spectral/time-frequency transform"
        disallowed = families - {FAMILY_SPECTRAL, FAMILY_PREPROCESS, FAMILY_UNKNOWN}
        if disallowed:
            return False, "raw-phase plan must not jump directly to feature/stat or combine ops"
    elif phase == PHASE_FEATURE:
        if FAMILY_FEATURE not in families:
            return False, "feature-phase plan must contain feature/stat ops"
        if len(op_names) < 2:
            return False, "feature-phase plan must use at least two distinct feature ops"
        for step in steps:
            op_name = str(step.get("op_name", "")).strip()
            if get_op_family(op_name) != FAMILY_FEATURE:
                return False, "feature-phase plan must only add feature/stat ops on transformed leaves"
            for parent_id in _plan_parent_ids(step):
                parent = state.dag_state.nodes.get(parent_id)
                if parent is None or parent.stage == "input":
                    return False, "feature-phase plan must target transformed leaves"
                if get_op_family(get_node_op_name(parent)) != FAMILY_SPECTRAL:
                    return False, "feature-phase plan must build on spectral/time-frequency transforms"
    elif phase == PHASE_COMBINE:
        if FAMILY_COMBINE not in families:
            return False, "combine-phase plan must contain a combine/cross op"

    return True, "ok"


def build_fallback_plan(state: PHMState, *, phase: str | None = None) -> list[Dict[str, Any]]:
    phase = phase or infer_builder_phase(state)
    leaves = [leaf for leaf in state.dag_state.leaves if leaf in state.dag_state.nodes]
    target_width = max(1, int(state.min_width))

    if phase == PHASE_RAW:
        input_leaves = [leaf for leaf in leaves if state.dag_state.nodes[leaf].stage == "input"]
        selected = input_leaves[: max(target_width, 2)]
        if not selected:
            return []
        steps: list[Dict[str, Any]] = []
        for index, parent_id in enumerate(selected):
            steps.append(
                {
                    "parent": parent_id,
                    "op_name": RAW_FALLBACK_TRANSFORMS[index % len(RAW_FALLBACK_TRANSFORMS)],
                    "params": {},
                }
            )
        return steps

    if phase == PHASE_FEATURE:
        transform_leaves = [
            leaf
            for leaf in leaves
            if state.dag_state.nodes[leaf].stage != "input"
            and get_op_family(get_node_op_name(state.dag_state.nodes[leaf])) == FAMILY_SPECTRAL
        ]
        selected = transform_leaves[: max(target_width, 2)]
        if not selected:
            return build_fallback_plan(state, phase=PHASE_RAW)
        steps = []
        for index, parent_id in enumerate(selected):
            steps.append(
                {
                    "parent": parent_id,
                    "op_name": FEATURE_FALLBACK_OPS[index % len(FEATURE_FALLBACK_OPS)],
                    "params": {},
                }
            )
        return steps

    feature_leaves = [leaf for leaf in leaves if leaf in state.dag_state.nodes and is_feature_leaf(state, leaf)]
    if len(feature_leaves) < 2:
        return build_fallback_plan(state, phase=PHASE_FEATURE)
    steps = []
    for index in range(0, len(feature_leaves) - 1, 2):
        pair = feature_leaves[index:index + 2]
        if len(pair) < 2:
            continue
        steps.append(
            {
                "parent": ",".join(pair),
                "op_name": COMBINE_FALLBACK_OPS[(index // 2) % len(COMBINE_FALLBACK_OPS)],
                "params": {},
            }
        )
    return steps


def evaluate_builder_richness(state: PHMState) -> Dict[str, Any]:
    processed_nodes = [node for node in state.dag_state.nodes.values() if node.stage != "input"]
    unique_ops = sorted({get_node_op_name(node) for node in processed_nodes if get_node_op_name(node)})
    families = sorted({get_op_family(op_name) for op_name in unique_ops if get_op_family(op_name) != FAMILY_UNKNOWN})
    processed_leaf_ids = [leaf for leaf in state.dag_state.leaves if leaf in state.dag_state.nodes and state.dag_state.nodes[leaf].stage != "input"]
    processed_leaf_families = {
        get_op_family(get_node_op_name(state.dag_state.nodes[leaf]))
        for leaf in processed_leaf_ids
        if get_op_family(get_node_op_name(state.dag_state.nodes[leaf])) != FAMILY_UNKNOWN
    }
    depth = get_dag_depth(state.dag_state)
    has_transform = FAMILY_SPECTRAL in families
    has_feature_stat = FAMILY_FEATURE in families
    is_root_only = not processed_leaf_ids
    is_mean_only = unique_ops == ["mean"]

    @lru_cache(maxsize=None)
    def _has_input_parent(node_id: str) -> bool:
        node = state.dag_state.nodes.get(node_id)
        if node is None:
            return False
        parent_ids = node.parents if isinstance(node.parents, list) else [node.parents]
        parent_ids = [str(parent_id).strip() for parent_id in parent_ids if str(parent_id).strip()]
        if not parent_ids:
            return node.stage == "input"
        for parent_id in parent_ids:
            parent = state.dag_state.nodes.get(parent_id)
            if parent is None:
                continue
            if parent.stage == "input":
                return True
            if _has_input_parent(parent_id):
                return True
        return False

    complete_strong_paths: list[str] = []
    for leaf_id in processed_leaf_ids:
        leaf = state.dag_state.nodes[leaf_id]
        if get_op_family(get_node_op_name(leaf)) != FAMILY_FEATURE:
            continue
        parent_ids = leaf.parents if isinstance(leaf.parents, list) else [leaf.parents]
        for parent_id in [str(parent_id).strip() for parent_id in parent_ids if str(parent_id).strip()]:
            parent = state.dag_state.nodes.get(parent_id)
            if parent is None:
                continue
            if get_op_family(get_node_op_name(parent)) != FAMILY_SPECTRAL:
                continue
            if not _has_input_parent(parent_id):
                continue
            complete_strong_paths.append(f"{parent_id}->{leaf_id}")
            break

    has_complete_strong_path = bool(complete_strong_paths)
    reasons: list[str] = []
    if depth < state.min_depth:
        reasons.append(f"depth {depth} is below min_depth {state.min_depth}")
    if len(processed_leaf_ids) < max(1, int(state.min_width)):
        reasons.append(
            f"processed terminal branches {len(processed_leaf_ids)} are below min_width {state.min_width}"
        )
    if not has_transform:
        reasons.append("missing spectral/time-frequency transform")
    if not has_feature_stat:
        reasons.append("missing feature/stat op")
    if len({family for family in families if family in {FAMILY_SPECTRAL, FAMILY_FEATURE, FAMILY_COMBINE}}) < 2:
        reasons.append("missing op family diversity")
    if not has_complete_strong_path:
        reasons.append("missing complete strong path input->spectral_transform->feature_stat")
    if is_root_only:
        reasons.append("root-only DAG")
    if is_mean_only:
        reasons.append("mean-only DAG")
    passes = not reasons
    return {
        "passes": passes,
        "reason": "ok" if passes else "; ".join(reasons),
        "depth": depth,
        "unique_ops": unique_ops,
        "families": families,
        "processed_leaf_count": len(processed_leaf_ids),
        "processed_leaf_ids": processed_leaf_ids,
        "processed_leaf_families": sorted(processed_leaf_families),
        "has_transform": has_transform,
        "has_spectral_transform": has_transform,
        "has_feature_stat": has_feature_stat,
        "has_complete_strong_path": has_complete_strong_path,
        "complete_strong_paths": complete_strong_paths,
        "is_root_only": is_root_only,
        "is_mean_only": is_mean_only,
    }


def has_hard_builder_errors(state: PHMState, *, extra_reason: str = "") -> bool:
    haystack: list[str] = list(state.dag_state.error_log)
    if extra_reason:
        haystack.append(extra_reason)
    combined = "\n".join(haystack).lower()
    return any(marker in combined for marker in HARD_ERROR_MARKERS) or get_dag_depth(state.dag_state) == -1


def reflection_fallback(state: PHMState, *, extra_reason: str = "") -> tuple[str, str]:
    quality = evaluate_builder_richness(state)
    if has_hard_builder_errors(state, extra_reason=extra_reason):
        reason = extra_reason or quality["reason"] or "builder encountered a structural error"
        return "halt", reason
    if quality["passes"]:
        return "finish", quality["reason"]
    if quality["depth"] < state.max_depth:
        reason = quality["reason"]
        if extra_reason:
            reason = f"{extra_reason}; {reason}" if reason != "ok" else extra_reason
        return "need_patch", reason
    reason = quality["reason"]
    if extra_reason:
        reason = f"{extra_reason}; {reason}" if reason != "ok" else extra_reason
    return "halt", reason


def quality_failure_message(state: PHMState) -> str:
    quality = evaluate_builder_richness(state)
    return f"Builder produced an invalid DAG: {quality['reason']}"
