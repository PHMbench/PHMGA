"""Hydra-backed runtime config composition for the paper-oriented repository."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import re

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config_root() -> Path:
    return _repo_root() / "config"


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _path_to_overrides(path: Path) -> list[str]:
    config_root = _config_root().resolve()
    resolved = path.resolve()
    if resolved == (config_root / "config.yaml").resolve():
        return []

    try:
        relative = resolved.relative_to(config_root)
    except ValueError as exc:
        raise ValueError(f"Config path {resolved} is outside {config_root}") from exc

    if not relative.suffix == ".yaml":
        raise ValueError(f"Config path must point to a .yaml file: {resolved}")

    group = relative.parts[0]
    name = relative.stem
    if group == "runs":
        return [f"+runs={name}"]
    if group == "data":
        return [f"data={name}"]
    if group == "experiment":
        return [f"experiment={name}"]
    if group == "model":
        return [f"model={name}"]
    raise ValueError(f"Unsupported config group for path-based loading: {resolved}")


def compose_runtime_config(
    config_input: Optional[Union[str, Path]] = None,
    *,
    overrides: Optional[Iterable[str]] = None,
) -> DictConfig:
    """Compose a Hydra config for scripts, tests, or the root `main.py`."""

    final_overrides = list(overrides or [])
    if config_input is not None:
        final_overrides.extend(_path_to_overrides(Path(config_input)))

    with initialize_config_dir(config_dir=str(_config_root()), version_base=None):
        return compose(config_name="config", overrides=final_overrides)


def to_runtime_dict(
    config_input: Union[DictConfig, Dict[str, Any]],
    *,
    output_dir: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Resolve Hydra/OmegaConf config into the plain dict expected by business code."""

    if isinstance(config_input, DictConfig):
        merged = OmegaConf.to_container(config_input, resolve=True)
    else:
        merged = deepcopy(config_input)
    if not isinstance(merged, dict):
        raise ValueError("Expected config composition to resolve into a mapping.")

    merged.setdefault("runtime", {})
    runtime = merged["runtime"]
    if not isinstance(runtime, dict):
        raise ValueError("Expected runtime config to be a mapping.")

    data_cfg = dict(merged.get("data", {}))
    experiment_cfg = dict(merged.get("experiment", {}))
    if "dataset_name" not in data_cfg:
        raise ValueError(
            "Runtime config must select a dataset via +runs=<preset>, data=<group>, or a config path under config/data or config/runs."
        )
    if "graph_path" not in experiment_cfg:
        raise ValueError(
            "Runtime config must select an experiment path via +runs=<preset>, experiment=<group>, or a config path under config/experiment or config/runs."
        )
    runtime["dataset_key"] = str(data_cfg.get("key", _slugify(str(data_cfg.get("dataset_name", "")))))
    runtime["dataset_name"] = data_cfg["dataset_name"]
    runtime["graph_path"] = experiment_cfg["graph_path"]
    runtime.setdefault("action", "run_case")

    if config_path is not None:
        resolved = Path(config_path).resolve()
        runtime["config_path"] = str(resolved)
        runtime["config_name"] = resolved.stem
    else:
        runtime.setdefault("config_path", "<hydra>")
        runtime.setdefault("config_name", "hydra_main")

    if output_dir:
        runtime["output_dir"] = output_dir
    elif not runtime.get("output_dir"):
        dataset_slug = _slugify(str(data_cfg["dataset_name"]))
        graph_slug = _slugify(str(experiment_cfg["graph_path"]))
        runtime["output_dir"] = str(_repo_root() / "artifacts" / f"{dataset_slug}_{graph_slug}")

    return merged


def load_runtime_config(
    config_input: Union[str, Path, DictConfig, Dict[str, Any]],
    *,
    output_dir: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Resolve a config path or Hydra DictConfig into the shared runtime dict."""

    if isinstance(config_input, DictConfig):
        return to_runtime_dict(config_input, output_dir=output_dir)
    if isinstance(config_input, dict):
        return to_runtime_dict(config_input, output_dir=output_dir)

    path = Path(config_input).resolve()
    composed = compose_runtime_config(path, overrides=overrides)
    return to_runtime_dict(composed, output_dir=output_dir, config_path=path)
