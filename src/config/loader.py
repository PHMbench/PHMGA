"""Configuration loader for the rebuilt paper-oriented repository."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Set

import re
import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and enforce a mapping-shaped top level."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested config groups while preserving scalar overrides."""
    merged = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_root(start_dir: Path) -> Path:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "data").is_dir() and (candidate / "experiment").is_dir():
            return candidate
    raise ValueError(f"Could not find config root above {start_dir}")


def _load_raw_config(path: Path, seen: Set[Path] | None = None) -> Dict[str, Any]:
    resolved = path.resolve()
    trail = seen or set()
    if resolved in trail:
        raise ValueError(f"Recursive base_config detected at {resolved}")
    trail = set(trail)
    trail.add(resolved)

    data = _load_yaml(resolved)
    base_ref = data.pop("base_config", None)
    if not base_ref:
        return data
    base_path = (resolved.parent / str(base_ref)).resolve()
    base = _load_raw_config(base_path, trail)
    return _deep_merge(base, data)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_runtime_config(
    config_path: str | Path,
    *,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Resolve the single runtime config used by scripts and tests."""
    root = Path(config_path).resolve()
    config_root = _resolve_config_root(root.parent)
    merged_raw = _load_raw_config(root)

    data_override = deepcopy(merged_raw.pop("data", {}))
    experiment_override = deepcopy(merged_raw.pop("experiment", {}))
    model_override = deepcopy(merged_raw.pop("model", {}))

    defaults = dict(merged_raw.get("defaults", {}))
    dataset_key = str(defaults.get("dataset", "")).strip().lower()
    graph_key = str(defaults.get("graph_path", "")).strip().lower()
    if not dataset_key:
        raise ValueError(f"Config {root} is missing defaults.dataset")
    if not graph_key:
        raise ValueError(f"Config {root} is missing defaults.graph_path")

    dataset_cfg = _load_yaml(config_root / "data" / f"{dataset_key}.yaml")
    experiment_cfg = _load_yaml(config_root / "experiment" / f"{graph_key}.yaml")
    model_cfg = _load_yaml(config_root / "model" / "default.yaml")

    merged = _deep_merge(merged_raw, dataset_cfg)
    merged = _deep_merge(merged, experiment_cfg)
    merged = _deep_merge(merged, model_cfg)
    merged = _deep_merge(merged, {"data": data_override, "experiment": experiment_override, "model": model_override})

    merged.setdefault("runtime", {})
    merged["runtime"]["dataset_key"] = dataset_key
    merged["runtime"]["dataset_name"] = merged["data"]["dataset_name"]
    merged["runtime"]["graph_path"] = merged["experiment"]["graph_path"]
    merged["runtime"]["config_path"] = str(root)
    merged["runtime"]["config_name"] = root.stem
    if output_dir:
        merged["runtime"]["output_dir"] = output_dir
    else:
        dataset_slug = _slugify(str(merged["data"]["dataset_name"]))
        graph_slug = _slugify(str(merged["experiment"]["graph_path"]))
        merged["runtime"].setdefault(
            "output_dir",
            str(config_root.parent / "artifacts" / f"{dataset_slug}_{graph_slug}"),
        )
    return merged
