from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_runtime_config(
    config_path: str | Path,
    *,
    dataset_name: str | None = None,
    graph_path: str | None = None,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    root = Path(config_path).resolve()
    config_dir = root.parent
    base = _load_yaml(root)

    dataset_key = str(dataset_name or base.get("defaults", {}).get("dataset", "rm101")).strip().lower()
    graph_key = str(graph_path or base.get("defaults", {}).get("graph_path", "dag_only")).strip().lower()

    dataset_cfg = _load_yaml(config_dir / "data" / f"{dataset_key}.yaml")
    experiment_cfg = _load_yaml(config_dir / "experiment" / f"{graph_key}.yaml")
    model_cfg = _load_yaml(config_dir / "model" / "default.yaml")

    merged = _deep_merge(base, dataset_cfg)
    merged = _deep_merge(merged, experiment_cfg)
    merged = _deep_merge(merged, model_cfg)
    merged.setdefault("runtime", {})
    merged["runtime"]["dataset_name"] = merged["data"]["dataset_name"]
    merged["runtime"]["graph_path"] = merged["experiment"]["graph_path"]
    if output_dir:
        merged["runtime"]["output_dir"] = output_dir
    else:
        merged["runtime"].setdefault(
            "output_dir",
            str(config_dir.parent / "artifacts" / f"{dataset_key}_{graph_key}"),
        )
    return merged
