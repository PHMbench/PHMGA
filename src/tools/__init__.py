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