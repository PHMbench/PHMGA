"""Pure-YAML layered config loader without Hydra."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "config"

RUN_ALIASES = {
    "case1": "rm101_ml_openrouter",
    "case_exp2": "rm101_ml_openrouter",
    "case_exp2.5": "rm101_torch_openrouter",
    "case_exp_ottawa": "ottawa_ml_gemini",
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return payload


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    resolved = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        value = _parse_override_value(raw_value)
        cursor = resolved
        parts = [part for part in dotted_key.split(".") if part]
        if not parts:
            raise ValueError(f"Invalid override key: {item}")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path is not a mapping: {dotted_key}")
        cursor[parts[-1]] = value
    return resolved


def resolve_run_name(run_name: str) -> str:
    normalized = RUN_ALIASES.get(run_name, run_name)
    preset_path = CONFIG_ROOT / "runs" / f"{normalized}.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Unknown run preset: {run_name}")
    return normalized


def load_runtime_config(
    run_name: str,
    *,
    action: str | None = None,
    output_dir: str | None = None,
    overrides: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Load one runtime config via base -> data preset -> experiment preset -> run preset."""

    resolved_run_name = resolve_run_name(run_name)
    base_cfg = _load_yaml(CONFIG_ROOT / "base.yaml")
    run_cfg = _load_yaml(CONFIG_ROOT / "runs" / f"{resolved_run_name}.yaml")

    data_preset = str(run_cfg.pop("data_preset"))
    experiment_preset = str(run_cfg.pop("experiment_preset"))

    data_cfg = _load_yaml(CONFIG_ROOT / "data" / f"{data_preset}.yaml")
    experiment_cfg = _load_yaml(CONFIG_ROOT / "experiment" / f"{experiment_preset}.yaml")

    config = _deep_merge(base_cfg, {"data": data_cfg})
    config = _deep_merge(config, {"experiment": experiment_cfg})
    config = _deep_merge(config, run_cfg)
    if overrides:
        config = _apply_overrides(config, overrides)

    runtime_cfg = dict(config.get("runtime", {}))
    runtime_cfg.setdefault("output_dir", str(ROOT / "artifacts" / resolved_run_name))
    if action is not None:
        runtime_cfg["action"] = action
    if output_dir is not None:
        runtime_cfg["output_dir"] = output_dir
    runtime_cfg["run_name"] = resolved_run_name
    runtime_cfg["requested_run_name"] = run_name
    config["runtime"] = runtime_cfg
    return config
