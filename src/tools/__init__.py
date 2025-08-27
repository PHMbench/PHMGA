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

# Import all schema files to trigger the registration of operators
from . import aggregate_schemas
from . import transform_schemas
from . import expand_schemas
from . import decision_schemas
from . import multi_schemas

# Import other tools that might be useful
# Note: comparator_tool import is deferred to avoid circular dependency
def compare_processed_nodes(*args, **kwargs):
    """Lazy import wrapper for compare_processed_nodes to avoid circular imports."""
    from .comparator_tool import compare_processed_nodes as _compare_processed_nodes
    return _compare_processed_nodes(*args, **kwargs)

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