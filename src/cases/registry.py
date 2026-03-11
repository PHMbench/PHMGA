from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List


CaseRunner = Callable[..., Any]

_CASE_REGISTRY: Dict[str, CaseRunner] = {}
_BUILTIN_CASES: Dict[str, str] = {
    "case1": "src.cases.case1:run_case",
}


def register_case_runner(name: str, runner: CaseRunner) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("case runner name cannot be empty")
    _CASE_REGISTRY[key] = runner


def get_case_runner(name: str) -> CaseRunner:
    key = str(name).strip()
    if key not in _CASE_REGISTRY and key in _BUILTIN_CASES:
        module_name, attr_name = _BUILTIN_CASES[key].split(":", 1)
        module = importlib.import_module(module_name)
        register_case_runner(key, getattr(module, attr_name))
    if key not in _CASE_REGISTRY:
        raise KeyError(f"Unknown case runner: {key}")
    return _CASE_REGISTRY[key]


def list_case_runners() -> List[str]:
    return sorted(set(_CASE_REGISTRY) | set(_BUILTIN_CASES))
