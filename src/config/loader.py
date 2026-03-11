from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping yaml at {path}, got {type(payload).__name__}.")
    return payload


def _normalize_config_target(
    config_path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    config_name: str | None = None,
) -> Tuple[Path, str]:
    if config_path is not None:
        path = Path(config_path).resolve()
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Expected yaml config path, got: {path}")
        return path.parent, path.stem
    if config_dir is None or not str(config_name or "").strip():
        raise ValueError("Either config_path or both config_dir and config_name must be provided.")
    return Path(config_dir).resolve(), str(config_name).strip()


def load_composed_config(
    config_path: str | Path | None = None,
    *,
    config_dir: str | Path | None = None,
    config_name: str | None = None,
    overrides: Iterable[str] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    normalized_dir, normalized_name = _normalize_config_target(
        config_path,
        config_dir=config_dir,
        config_name=config_name,
    )
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(normalized_dir), version_base=None):
        cfg = compose(config_name=normalized_name, overrides=list(overrides or []))
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected composed config to be a mapping, got {type(payload).__name__}.")
    return payload, [str((normalized_dir / f"{normalized_name}.yaml").resolve())]
