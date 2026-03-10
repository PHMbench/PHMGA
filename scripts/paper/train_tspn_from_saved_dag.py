#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.states.phm_states import DAGState, DataSetNode, InputData, PHMState, ProcessedData
from src.utils import initialize_state, initialize_state_vibench, save_state


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_source_mode(config: Dict[str, Any]) -> str:
    data_cfg = dict(config.get("data") or {})
    source_mode = str(data_cfg.get("source_mode") or "").strip().lower()
    if source_mode in {"fixed_ids", "vibench"}:
        return source_mode
    backend = str(data_cfg.get("backend") or "").strip().lower()
    return "vibench" if backend == "vibench" else "fixed_ids"


def _deserialize_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    stage = str(node.get("stage") or "processed")
    if stage == "input":
        return InputData(**node)
    if stage == "dataset":
        return DataSetNode(**node)
    return ProcessedData(**node)


def _load_dag_state(path: Path) -> DAGState:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid DAG JSON: {path}. Expected full DAGState JSON with channels/nodes/leaves; "
            "compact export_json() graph is not supported."
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError("DAG JSON must be a JSON object.")
    required = {"channels", "nodes", "leaves"}
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(
            f"DAG JSON must contain {sorted(required)}; missing={sorted(missing)}. "
            "export_json() compact graph is not supported for training input."
        )
    channels = list(raw.get("channels") or [])
    leaves = list(raw.get("leaves") or [])
    nodes_raw = dict(raw.get("nodes") or {})
    nodes = {str(node_id): _deserialize_node(node_obj) for node_id, node_obj in nodes_raw.items()}
    return DAGState(
        user_instruction=str(raw.get("user_instruction") or ""),
        channels=channels,
        nodes=nodes,
        leaves=leaves,
        error_log=list(raw.get("error_log") or []),
        graph_path=raw.get("graph_path"),
    )


def _init_state_from_case(config: Dict[str, Any]) -> PHMState:
    source_mode = _resolve_source_mode(config)
    data_cfg = dict(config.get("data") or {})
    data_cfg["source_mode"] = source_mode
    model_cfg = dict(config.get("model") or {})
    data_cfg.setdefault("model_profile", model_cfg.get("profile"))
    data_cfg.setdefault("autofit_dims", _parse_bool(model_cfg.get("autofit_dims"), True))
    data_cfg.setdefault("autofit_num_classes", _parse_bool(model_cfg.get("autofit_num_classes"), True))
    data_cfg.setdefault("model_config_path", model_cfg.get("config_path") or config.get("model_config_path"))

    train_backend = str(config.get("train_backend") or "tspn")
    allow_labels = bool(config.get("allow_test_labels_for_reporting", False))
    model_path = str(config.get("model_config_path") or model_cfg.get("config_path") or "")
    if source_mode == "vibench":
        return initialize_state_vibench(
            user_instruction=str(config.get("user_instruction") or ""),
            case_name=str(config.get("name") or "dag_resume"),
            data_cfg=data_cfg,
            allow_test_labels_for_reporting=allow_labels,
            train_backend=train_backend,
            model_config_path=model_path,
            save_dir=str(config.get("save_dir") or ""),
        )

    metadata_path = str(config.get("metadata_path") or data_cfg.get("metadata_path") or "")
    h5_path = str(config.get("h5_path") or data_cfg.get("h5_path") or "")
    ref_ids = list(config.get("ref_ids") or data_cfg.get("ref_ids") or [])
    test_ids = list(config.get("test_ids") or data_cfg.get("test_ids") or [])
    return initialize_state(
        user_instruction=str(config.get("user_instruction") or ""),
        metadata_path=metadata_path,
        h5_path=h5_path,
        ref_ids=ref_ids,
        test_ids=test_ids,
        case_name=str(config.get("name") or "dag_resume"),
        allow_test_labels_for_reporting=allow_labels,
        train_backend=train_backend,
        model_config_path=model_path or None,
        save_dir=str(config.get("save_dir") or ""),
        data_cfg=data_cfg,
    )


def _run_cmd(cmd: list[str], *, cwd: Path, env: Dict[str, str], log_path: Path, dry_run: bool) -> int:
    print(f"[cmd] {' '.join(cmd)}")
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            out.flush()
            print(line, end="")
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume TSPN training from saved DAG/state.")
    parser.add_argument("--case-config", required=True, help="Base case yaml path.")
    parser.add_argument("--state-pkl", default="", help="Existing built_state.pkl path.")
    parser.add_argument("--dag-json", default="", help="Full DAGState JSON path (channels/nodes/leaves).")
    parser.add_argument("--output-root", default="", help="Output root; default uses case save_dir.")
    parser.add_argument("--run-name", default="", help="Optional run suffix.")
    parser.add_argument("--preflight", action="store_true", help="Run preflight before case1.")
    parser.add_argument("--allow-unverified-state", action="store_true", help="Allow state pkl without .sha256.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    if bool(args.state_pkl) == bool(args.dag_json):
        raise SystemExit("Exactly one of --state-pkl or --dag-json must be provided.")

    repo_root = REPO_ROOT
    case_path = Path(args.case_config).resolve()
    if not case_path.exists():
        raise SystemExit(f"Case config not found: {case_path}")
    base_cfg = _load_yaml(case_path)

    run_name = str(args.run_name).strip() or time.strftime("%Y%m%d-%H%M%S")
    base_name = str(base_cfg.get("name") or "dag_resume")
    case_name = f"{base_name}__dag_resume__{run_name}"

    save_root = Path(args.output_root).resolve() if args.output_root else Path(str(base_cfg.get("save_dir") or repo_root / "save")).resolve()
    case_dir = save_root / case_name
    resolved_state = case_dir / "built_state.pkl"
    resolved_cfg = save_root / "_resolved_cases" / f"{case_name}.yaml"
    log_dir = save_root / "_logs" / case_name

    resolved_case_cfg = dict(base_cfg)
    resolved_case_cfg["name"] = case_name
    resolved_case_cfg["save_dir"] = str(save_root)
    resolved_case_cfg["state_save_path"] = str(resolved_state)
    resolved_case_cfg["report_path"] = str(case_dir / "final_report.md")
    resolved_case_cfg["run_executor"] = True
    resolved_case_cfg["train_backend"] = "tspn"
    data_cfg = dict(resolved_case_cfg.get("data") or {})
    data_cfg["use_dag_model_config"] = True
    resolved_case_cfg["data"] = data_cfg
    _dump_yaml(resolved_cfg, resolved_case_cfg)

    if args.dry_run:
        if args.state_pkl:
            src = Path(args.state_pkl).resolve()
            if not src.exists():
                raise SystemExit(f"state pkl not found: {src}")
        else:
            dag_path = Path(args.dag_json).resolve()
            if not dag_path.exists():
                raise SystemExit(f"dag json not found: {dag_path}")
            _load_dag_state(dag_path)  # format validation only
        print("[dry-run] skip state materialization")
    elif args.state_pkl:
        src = Path(args.state_pkl).resolve()
        if not src.exists():
            raise SystemExit(f"state pkl not found: {src}")
        resolved_state.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, resolved_state)
        src_sha = Path(str(src) + ".sha256")
        dst_sha = Path(str(resolved_state) + ".sha256")
        if src_sha.exists():
            shutil.copy2(src_sha, dst_sha)
        elif not args.allow_unverified_state:
            raise SystemExit(
                f"Missing checksum: {src_sha}. Use --allow-unverified-state for local debug."
            )
    else:
        dag_path = Path(args.dag_json).resolve()
        if not dag_path.exists():
            raise SystemExit(f"dag json not found: {dag_path}")
        state = _init_state_from_case(resolved_case_cfg)
        state.dag_state = _load_dag_state(dag_path)
        state.train_backend = "tspn"
        state.save_dir = str(save_root)
        state.model_config_path = (
            resolved_case_cfg.get("model_config_path")
            or dict(resolved_case_cfg.get("model") or {}).get("config_path")
            or state.model_config_path
        )
        ok = save_state(state, str(resolved_state))
        if not ok:
            raise SystemExit("Failed to save reconstructed state.")

    env = dict(os.environ)
    if args.allow_unverified_state:
        env["PHM_ALLOW_UNVERIFIED_STATE"] = "1"

    if args.preflight:
        pre_cmd = [sys.executable, "main.py", "preflight", "--config", str(resolved_cfg)]
        pre_rc = _run_cmd(pre_cmd, cwd=repo_root, env=env, log_path=log_dir / "preflight.log", dry_run=args.dry_run)
        if pre_rc != 0:
            return pre_rc

    run_cmd = [sys.executable, "main.py", "case1", "--config", str(resolved_cfg)]
    run_rc = _run_cmd(run_cmd, cwd=repo_root, env=env, log_path=log_dir / "run.log", dry_run=args.dry_run)
    print(f"[resolved_case] {resolved_cfg}")
    print(f"[state] {resolved_state}")
    return run_rc


if __name__ == "__main__":
    raise SystemExit(main())
