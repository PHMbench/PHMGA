from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from src.schemas.config_schema import ResolvedConfig
from omegaconf import OmegaConf

from .data import normalize_runtime_config
from .loader import load_composed_config


def _apply_overrides(payload: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = OmegaConf.create(payload)
    for path, value in overrides.items():
        OmegaConf.update(merged, str(path), value, merge=True)
    rendered = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(rendered, dict):
        raise ValueError(f"Expected resolved override payload to be a mapping, got {type(rendered).__name__}.")
    return rendered


def resolve_config(
    config_path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    config_name: str | None = None,
    hydra_overrides: Iterable[str] | None = None,
    overrides: Dict[str, Any] | None = None,
) -> ResolvedConfig:
    payload, sources = load_composed_config(
        config_path,
        config_dir=config_dir,
        config_name=config_name,
        overrides=hydra_overrides,
    )
    if overrides:
        payload = _apply_overrides(payload, overrides)
    payload = normalize_runtime_config(dict(payload))
    payload.setdefault("metadata", {})
    payload["sources"] = list(sources)
    return ResolvedConfig.model_validate(payload)


def write_resolved_config(resolved: ResolvedConfig, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resolved.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_resolved_config_yaml(resolved: ResolvedConfig, output_path: str | Path) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(resolved.model_dump(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return out_path
