"""
This package contains all signal processing operators and tools.

Importing the modules here ensures that all operators decorated with @register_op
are correctly placed into the global OP_REGISTRY.
"""
from __future__ import annotations

# Core components for the operator registry
from .signal_processing_schemas import (
    OP_REGISTRY,
    PHMOperator,
    get_operator,
    AggregateOp,
    TransformOp,
    ExpandOp,
    DecisionOp,
    MultiVariableOp,
)

# Best-effort import schema files to trigger operator registration.
# Some schema modules may have optional heavy dependencies (e.g. skimage).
# We must not fail import of the whole package if one optional dependency is missing.
_SCHEMA_MODULES = [
    "aggregate_schemas",
    "transform_schemas",
    "expand_schemas",
    "decision_schemas",
    "multi_schemas",
]

for _mod in _SCHEMA_MODULES:  # pragma: no cover
    try:
        __import__(f"{__name__}.{_mod}")
    except Exception:
        # Optional dependency missing or import-time error; skip registration for that module.
        pass

# Import other tools that might be useful
from .comparator_tool import compare_processed_nodes

# Define what is exposed when a user does 'from src.tools import *'
__all__ = [
    "OP_REGISTRY",
    "PHMOperator",
    "get_operator",
    "compare_processed_nodes",
    "AggregateOp",
    "TransformOp",
    "ExpandOp",
    "DecisionOp",
    "MultiVariableOp",
]


if __name__ == "__main__":
    print("--- Testing tools package ---")
    # Ensure that operators from submodules are registered
    assert "mean" in OP_REGISTRY, "MeanOp should be registered"
    MeanOp = get_operator("mean")
    import numpy as np

    dummy = np.ones((1, 4, 1))
    mean_result = MeanOp().execute(dummy)
    assert mean_result.shape == (1, 1)

    print("\n--- tools package tests passed! ---")
