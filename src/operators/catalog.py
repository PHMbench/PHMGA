"""Operator catalog assembly for the paper-oriented workflow.

Concrete operator implementations live in category-specific modules. This file
only assembles them into one closed-world catalog used by planner, executor,
bridge, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .aggregate_ops import get_aggregate_operators
from .base import BaseIsomorphicOperator, OperatorSpec
from .decision_ops import get_decision_operators
from .expand_ops import get_expand_operators
from .multi_ops import get_multi_operators
from .transform_ops import get_transform_operators


def _build_operator_groups() -> Dict[str, Tuple[BaseIsomorphicOperator, ...]]:
    """Return the sanctioned operator groups keyed by schema category."""

    return {
        "EXPAND": get_expand_operators(),
        "TRANSFORM": get_transform_operators(),
        "AGGREGATE": get_aggregate_operators(),
        "MULTI_VARIABLE": get_multi_operators(),
        "DECISION": get_decision_operators(),
    }


def _build_operator_index(
    groups: Dict[str, Tuple[BaseIsomorphicOperator, ...]],
) -> Dict[str, BaseIsomorphicOperator]:
    """Build the closed operator lookup indexed by `op_uid`."""

    operators: Dict[str, BaseIsomorphicOperator] = {}
    for instances in groups.values():
        for operator in instances:
            operators[operator.spec.op_uid] = operator
    return operators


def _build_plan_name_aliases() -> Dict[str, str]:
    """Map planner-facing short names to catalog op_uids."""

    return {
        "normalize": "signal.normalize",
        "fft": "signal.fft_mag",
        "fft_mag": "signal.fft_mag",
        "stft": "signal.stft",
        "patch": "signal.patch",
        "filter": "signal.filter",
        "hilbert_envelope": "signal.hilbert_envelope",
        "psd": "signal.psd",
        "mean": "feature.mean",
        "std": "feature.std",
        "rms": "feature.rms",
        "kurtosis": "feature.kurtosis",
        "crest_factor": "feature.crest_factor",
        "band_power": "feature.band_power",
        "spectral_centroid": "feature.spectral_centroid",
        "concatenate": "multi.concatenate",
        "cross_correlation": "multi.cross_correlation",
        "threshold": "decision.threshold",
        "decision.threshold": "decision.threshold",
    }


@dataclass
class OperatorCatalog:
    """Lookup table that exposes the sanctioned operator subset."""

    operators: Dict[str, BaseIsomorphicOperator]
    plan_name_aliases: Dict[str, str]

    def get(self, op_uid: str) -> BaseIsomorphicOperator:
        if op_uid not in self.operators:
            raise KeyError(f"Unknown operator: {op_uid}")
        return self.operators[op_uid]

    def resolve_plan_name(self, op_name: str) -> str:
        normalized = op_name.strip().lower()
        if normalized in self.plan_name_aliases:
            return self.plan_name_aliases[normalized]
        if normalized in self.operators:
            return normalized
        raise KeyError(f"Unknown plan operator name: {op_name}")

    def get_by_plan_name(self, op_name: str) -> BaseIsomorphicOperator:
        return self.get(self.resolve_plan_name(op_name))

    def specs(self) -> List[OperatorSpec]:
        return [operator.spec for operator in self.operators.values()]

    def feature_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.schema_category == "AGGREGATE"]

    def transform_ops(self) -> List[str]:
        return [spec.op_uid for spec in self.specs() if spec.schema_category in {"TRANSFORM", "EXPAND"}]

    def build_summary_rows(self) -> List[Dict[str, Any]]:
        """Return the prompt-safe summary rows used by planner and executor."""

        rows: list[dict[str, Any]] = []
        for spec in self.specs():
            rows.append(
                {
                    "op_uid": spec.op_uid,
                    "op_name": spec.op_name,
                    "name": spec.name,
                    "schema_category": spec.schema_category,
                    "rank_class": spec.rank_class,
                    "description": spec.description,
                    "input_spec": spec.input_spec,
                    "output_spec": spec.output_spec,
                    "input_shape_rule": spec.input_shape_rule,
                    "output_shape_rule": spec.output_shape_rule,
                    "execution_role": spec.execution_role,
                    "legal_paths": spec.legal_paths,
                    "param_schema": spec.param_schema,
                    "param_defaults": spec.param_defaults,
                    "param_docs": spec.param_docs,
                    "llm_tunable_params": spec.llm_tunable_params,
                    "planning_notes": spec.planning_notes,
                }
            )
        return rows

    def summary(self) -> List[Dict[str, Any]]:
        """Prompt-safe summary with C_Agent-style schema semantics."""

        return self.build_summary_rows()


def get_operator_catalog() -> OperatorCatalog:
    """Build the closed-world operator catalog used by scripts and tests."""

    groups = _build_operator_groups()
    return OperatorCatalog(
        operators=_build_operator_index(groups),
        plan_name_aliases=_build_plan_name_aliases(),
    )


__all__ = [
    "OperatorCatalog",
    "_build_operator_groups",
    "_build_operator_index",
    "_build_plan_name_aliases",
    "get_operator_catalog",
]
