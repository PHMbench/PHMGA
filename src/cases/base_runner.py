from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

from src.config import resolve_config, write_resolved_config, write_resolved_config_yaml

from .helpers import build_case_paths, write_metadata_snapshot
from .registry import get_case_runner


def _stringify_overrides(overrides: Iterable[str] | None) -> list[str]:
    return [str(item).strip() for item in list(overrides or []) if str(item).strip()]


def _select_case_name(resolved_payload: Dict[str, Any], explicit_case_name: str | None = None) -> str:
    if explicit_case_name:
        return str(explicit_case_name).strip()
    cases_cfg = dict(resolved_payload.get("cases") or {})
    selected = str(cases_cfg.get("selected") or resolved_payload.get("name") or "case1").strip()
    if not selected:
        raise ValueError("Unable to resolve case runner name from config.")
    return selected


def _resolve_save_root(resolved_payload: Dict[str, Any], save_root: str | Path | None = None) -> Path:
    if save_root is not None:
        return Path(save_root).resolve()
    candidate = resolved_payload.get("save_dir")
    if candidate:
        return Path(str(candidate)).resolve()
    system_paths = dict((resolved_payload.get("system") or {}).get("paths") or {})
    if system_paths.get("save_root"):
        return Path(str(system_paths["save_root"])).resolve()
    return (Path.cwd() / "save").resolve()


def prepare_case_runtime(
    *,
    config_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    config_name: str | None = None,
    hydra_overrides: Iterable[str] | None = None,
    case_name: str | None = None,
    save_root: str | Path | None = None,
    metadata_snapshot: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    resolved = resolve_config(
        config_path,
        config_dir=config_dir,
        config_name=config_name,
        hydra_overrides=_stringify_overrides(hydra_overrides),
    )
    resolved_payload = resolved.model_dump()
    selected_case = _select_case_name(resolved_payload, case_name)
    resolved_save_root = _resolve_save_root(resolved_payload, save_root)
    paths = build_case_paths(selected_case, resolved_save_root)
    write_resolved_config(resolved, paths["resolved_config_path"])
    runtime_config_path = Path(paths["case_dir"]) / "resolved_runtime.yaml"
    write_resolved_config_yaml(resolved, runtime_config_path)
    write_metadata_snapshot(
        metadata_snapshot or {"row_count": 0, "columns": []},
        paths["metadata_snapshot_path"],
    )
    graph_cfg = dict(resolved_payload.get("graphs") or {})
    runtime_manifest_path = Path(paths["case_dir"]) / "runtime_manifest.json"
    runtime_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest_path.write_text(
        json.dumps(
            {
                "selected_case": selected_case,
                "selected_graph": str(graph_cfg.get("selected") or ""),
                "builder_graph": str(graph_cfg.get("builder_name") or ""),
                "executor_graph": str(graph_cfg.get("executor_name") or ""),
                "config_dir": str(Path(config_dir).resolve()) if config_dir else None,
                "config_name": config_name,
                "hydra_overrides": _stringify_overrides(hydra_overrides),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = dict(paths)
    out["runtime_config_path"] = str(runtime_config_path)
    out["runtime_manifest_path"] = str(runtime_manifest_path)
    out["selected_case"] = selected_case
    out["selected_graph"] = str(graph_cfg.get("selected") or "")
    out["builder_graph"] = str(graph_cfg.get("builder_name") or "")
    out["executor_graph"] = str(graph_cfg.get("executor_name") or "")
    return out


def run_registered_case(
    *,
    config_path: str | Path | None = None,
    config_dir: str | Path | None = None,
    config_name: str | None = None,
    hydra_overrides: Iterable[str] | None = None,
    case_name: str | None = None,
    save_root: str | Path | None = None,
) -> Dict[str, str]:
    runtime = prepare_case_runtime(
        config_path=config_path,
        config_dir=config_dir,
        config_name=config_name,
        hydra_overrides=hydra_overrides,
        case_name=case_name,
        save_root=save_root,
    )
    runner = get_case_runner(runtime["selected_case"])
    runner(str(runtime["runtime_config_path"]))
    return runtime
