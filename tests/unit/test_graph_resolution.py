from __future__ import annotations

import pytest

from main import run_case
from src.phm_outer_graph import resolve_graph


def test_resolve_graph_accepts_known_names():
    assert resolve_graph("builder") is not None
    assert resolve_graph("executor") is not None
    assert resolve_graph("with_report") is not None


def test_resolve_graph_rejects_unknown_name():
    with pytest.raises(ValueError):
        resolve_graph("not_a_graph")


def test_executor_requires_existing_state_file(make_case_config):
    case = make_case_config(case_name="case_executor_missing_state", graph="executor")

    with pytest.raises(FileNotFoundError):
        run_case(case["case_name"], config_root=case["config_root"])
