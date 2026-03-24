"""Case-based YAML config loading for the graph-first PHMGA entrypoint."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "config"
VALID_GRAPHS = {"builder", "executor", "with_report"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def resolve_case_path(case_name: str, *, config_root: str | Path | None = None) -> Path:
    root = Path(config_root) if config_root is not None else CONFIG_ROOT
    path = root / f"{case_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown case config: {case_name}")
    return path


def normalize_case_config(config: Dict[str, Any], *, case_name: str) -> Dict[str, Any]:
    resolved = deepcopy(config)
    builder_cfg = dict(resolved.get("builder") or {})
    builder_cfg.setdefault("graph", "with_report")
    builder_cfg.setdefault("min_depth", 4)
    builder_cfg.setdefault("min_width", 2)
    builder_cfg.setdefault("max_depth", 8)
    graph_name = str(builder_cfg["graph"]).strip()
    if graph_name not in VALID_GRAPHS:
        raise ValueError(
            f"Invalid builder.graph={graph_name!r}; expected one of {sorted(VALID_GRAPHS)}."
        )
    resolved["builder"] = builder_cfg
    resolved.setdefault("name", case_name)
    resolved.setdefault("llm", {})
    return resolved


def load_case_config(case_name: str, *, config_root: str | Path | None = None) -> Dict[str, Any]:
    path = resolve_case_path(case_name, config_root=config_root)
    return normalize_case_config(_load_yaml(path), case_name=case_name)


__all__ = [
    "CONFIG_ROOT",
    "VALID_GRAPHS",
    "load_case_config",
    "normalize_case_config",
    "resolve_case_path",
]
