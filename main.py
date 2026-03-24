"""Case-based CLI for the graph-first PHMGA workflow."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

from src.builder_quality import evaluate_builder_richness, quality_failure_message
from src.config import load_case_config, resolve_case_path
from src.phm_outer_graph import resolve_graph
from src.states.phm_states import PHMState
from src.utils import generate_final_report, get_dag_depth, initialize_state, load_state, save_state


def _artifact_root(config: dict) -> str:
    save_dir = str(config.get("save_dir", "")).strip()
    if save_dir:
        return str(Path(save_dir).expanduser().resolve())
    report_path = str(config.get("report_path", "")).strip()
    if report_path:
        return str(Path(report_path).expanduser().resolve().parent)
    state_path = str(config.get("state_save_path", "")).strip()
    if state_path:
        return str(Path(state_path).expanduser().resolve().parent)
    return str((Path.cwd() / "artifacts").resolve())


def _runtime_context(config: dict) -> dict:
    return {
        "llm": deepcopy(dict(config.get("llm") or {})),
        "runtime": {
            "graph": str(config["builder"]["graph"]),
            "output_dir": _artifact_root(config),
            "run_name": str(config["name"]),
        },
    }


def _configure_state(state: PHMState, config: dict) -> PHMState:
    builder_cfg = dict(config.get("builder") or {})
    state.case_name = str(config["name"])
    state.user_instruction = str(config["user_instruction"])
    state.min_depth = int(builder_cfg.get("min_depth", state.min_depth))
    state.min_width = int(builder_cfg.get("min_width", state.min_width))
    state.max_depth = int(builder_cfg.get("max_depth", state.max_depth))
    state.runtime_config = _runtime_context(config)
    return state


def _fresh_state(config: dict) -> PHMState:
    state = initialize_state(
        user_instruction=str(config["user_instruction"]),
        metadata_path=str(config["metadata_path"]),
        h5_path=str(config["h5_path"]),
        ref_ids=list(config["ref_ids"]),
        test_ids=list(config["test_ids"]),
        case_name=str(config["name"]),
        use_window=bool(config.get("use_window", True)),
    )
    return _configure_state(state, config)


def _apply_updates(state: PHMState, updates: dict) -> None:
    if "dag_state" in updates:
        state._tracker_instance = None
    for key, value in updates.items():
        if key in type(state).model_fields:
            setattr(state, key, value)


def _run_graph(app, state: PHMState) -> PHMState:
    current = state.model_copy(deep=True)
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    for event in app.stream(current, config=thread_config):
        for _node_name, state_update in event.items():
            if state_update is None:
                continue
            if isinstance(state_update, PHMState):
                current = state_update
                continue
            if isinstance(state_update, dict):
                _apply_updates(current, state_update)
    return current


def _require_fields(config: dict, fields: Iterable[str]) -> None:
    missing = [field for field in fields if field not in config or config.get(field) in (None, "")]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")


def run_case(case_name: str, *, config_root: str | Path | None = None) -> dict:
    config_path = resolve_case_path(case_name, config_root=config_root)
    config = load_case_config(case_name, config_root=config_root)
    graph_name = str(config["builder"]["graph"])
    state_path = str(config.get("state_save_path", "")).strip()
    report_path = str(config.get("report_path", "")).strip()

    _require_fields(
        config,
        ["name", "metadata_path", "h5_path", "ref_ids", "test_ids", "state_save_path", "report_path"],
    )

    if graph_name == "with_report":
        state = _fresh_state(config)
    elif graph_name == "builder":
        loaded = load_state(state_path) if Path(state_path).exists() else None
        state = _configure_state(loaded, config) if loaded is not None else _fresh_state(config)
    elif graph_name == "executor":
        if not Path(state_path).exists():
            raise FileNotFoundError(f"Executor graph requires an existing state file: {state_path}")
        loaded = load_state(state_path)
        if loaded is None:
            raise RuntimeError(f"Failed to load state file: {state_path}")
        state = _configure_state(loaded, config)
    else:  # pragma: no cover - guarded by config validation + resolve_graph
        raise ValueError(f"Unknown graph {graph_name!r}")

    final_state = _run_graph(resolve_graph(graph_name), state)
    save_state(final_state, state_path)

    if graph_name == "builder":
        quality = evaluate_builder_richness(final_state)
        if not quality["passes"]:
            raise RuntimeError(quality_failure_message(final_state))

    if graph_name in {"with_report", "executor"}:
        if not str(final_state.final_report).strip():
            raise RuntimeError(f"Graph {graph_name!r} completed without producing final_report.")
        generate_final_report(final_state, report_path)

    return {
        "status": "ok",
        "case_name": case_name,
        "config_path": str(config_path),
        "graph": graph_name,
        "state_save_path": state_path,
        "report_path": report_path if graph_name in {"with_report", "executor"} else None,
        "dag_depth": get_dag_depth(final_state.dag_state),
        "dag_nodes": len(final_state.dag_state.nodes),
        "has_final_report": bool(str(final_state.final_report).strip()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a PHMGA case config from config/<case>.yaml.")
    parser.add_argument("case_name", help="Case config name under config/, for example case_exp_ottawa.")
    args = parser.parse_args(argv)

    if load_dotenv is not None and DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH)
    try:
        payload = run_case(args.case_name)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
ROOT = Path(__file__).resolve().parent
DOTENV_PATH = ROOT / ".env"
