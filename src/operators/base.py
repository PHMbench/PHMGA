"""Base contracts for the unified operator system.

The rebuilt repository keeps operator metadata intentionally explicit because
the same schema must satisfy four consumers at once:

- planner prompts
- executor-side parameter resolution
- validated DAG JSON export
- graph-dependent reporting
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import numpy as np
from pydantic import BaseModel, Field


BackendAvailability = Literal["np", "pt", "sym"]
ExecutionRole = Literal["trainable", "fixed", "proxy", "outer_only"]
GraphPath = Literal["dag_only", "ml", "torch"]
SchemaCategory = Literal["EXPAND", "TRANSFORM", "AGGREGATE", "MULTI_VARIABLE", "DECISION"]


class OperatorSpec(BaseModel):
    """Static metadata required by workflow, bridge, and reporting."""

    op_uid: str
    name: str
    schema_category: SchemaCategory
    description: str
    param_schema: Dict[str, str] = Field(default_factory=dict)
    param_defaults: Dict[str, Any] = Field(default_factory=dict)
    param_docs: Dict[str, str] = Field(default_factory=dict)
    input_shape_rule: str
    output_shape_rule: str
    backend_availability: List[BackendAvailability]
    execution_role: ExecutionRole
    legal_paths: List[GraphPath] = Field(default_factory=lambda: ["dag_only", "ml", "torch"])
    planning_notes: str = ""
    llm_tunable_params: List[str] = Field(default_factory=list)


class BaseIsomorphicOperator:
    """One semantic operator with multiple backend execution surfaces."""

    spec: OperatorSpec

    def forward_np(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def forward_pt(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_sym(self, x_sym: str, **kwargs: Any) -> str:
        raise NotImplementedError
