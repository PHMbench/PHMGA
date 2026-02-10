from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.explainable.operator_catalog import FEATURE_CATALOG, OPERATOR_CATALOG
from src.tools import OP_REGISTRY


def test_registered_ops_are_explicitly_cataloged():
    missing = []
    for op_name in sorted(OP_REGISTRY.keys()):
        if op_name not in OPERATOR_CATALOG and op_name not in FEATURE_CATALOG:
            missing.append(op_name)
    assert not missing, f"Operator catalog missing entries: {missing}"


def test_catalog_status_values_are_valid():
    valid = {"supported", "proxy", "unsupported"}
    for mapping in OPERATOR_CATALOG.values():
        assert mapping.status in valid
    for mapping in FEATURE_CATALOG.values():
        assert mapping.status in valid
