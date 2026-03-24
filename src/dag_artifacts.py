from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .states.phm_states import DAGTracker, PHMState


DATA_DIR = os.environ.get("PHM_DATA_DIR", os.path.join(os.getcwd(), "artifacts"))


def resolve_artifact_root(runtime_config: Dict[str, Any] | None, case_name: str) -> Path:
    runtime_config = runtime_config or {}
    runtime_output_dir = str(runtime_config.get("runtime", {}).get("output_dir", "")).strip()
    if runtime_output_dir:
        return Path(runtime_output_dir) / "_intermediate"
    base_save_dir = os.environ.get("PHM_SAVE_DIR", DATA_DIR)
    return Path(base_save_dir) / case_name


def export_tracker_artifacts(
    tracker: DAGTracker,
    *,
    output_dir: str | Path,
    stem: str = "dag",
    max_nodes: int | None = None,
    save_png: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    base = target_dir / stem

    result: Dict[str, Any] = {
        "json_path": None,
        "png_path": None,
        "dot_path": None,
        "warnings": [],
    }

    if save_json:
        result["json_path"] = tracker.write_json(str(base.with_suffix(".json")), max_nodes=max_nodes)

    if save_png:
        try:
            result["png_path"] = tracker.write_png(str(base.with_suffix(".png")))
        except Exception as exc:
            result["warnings"].append(f"png export failed: {exc}")
            try:
                result["dot_path"] = tracker.write_dot(str(base.with_suffix(".dot")))
            except Exception as dot_exc:
                result["warnings"].append(f"dot export failed: {dot_exc}")

    return result


def export_state_artifacts(
    state: PHMState,
    *,
    output_dir: str | Path,
    stem: str = "dag",
    max_nodes: int | None = None,
    save_png: bool = True,
    save_json: bool = True,
) -> Dict[str, Any]:
    return export_tracker_artifacts(
        state.tracker(),
        output_dir=output_dir,
        stem=stem,
        max_nodes=max_nodes,
        save_png=save_png,
        save_json=save_json,
    )
