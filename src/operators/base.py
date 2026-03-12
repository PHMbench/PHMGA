from __future__ import annotations

from typing import Any, Dict, List, Literal

import numpy as np
from pydantic import BaseModel, Field


BackendAvailability = Literal["np", "pt", "sym"]
ExecutionRole = Literal["trainable", "fixed", "proxy", "outer_only"]


class OperatorSpec(BaseModel):
    op_uid: str
    name: str
    param_schema: Dict[str, str] = Field(default_factory=dict)
    input_shape_rule: str
    output_shape_rule: str
    backend_availability: List[BackendAvailability]
    execution_role: ExecutionRole


class BaseIsomorphicOperator:
    spec: OperatorSpec

    def forward_np(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def forward_pt(self, x: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def forward_sym(self, x_sym: str, **kwargs: Any) -> str:
        raise NotImplementedError
