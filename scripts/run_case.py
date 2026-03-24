from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DOTENV_PATH = ROOT / ".env"

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

from main import run_case as run_graph_case
from src.config import load_case_config, resolve_case_path
from src.evaluation import summarize_dag_state
from src.manual_workflow import export_dag_artifacts
from src.utils import load_state


DEFAULT_ARTIFACT_ROOT = Path.cwd() / "artifacts"


def _workspace_root(case_name: str, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir).expanduser().resolve()
    DEFAULT_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{case_name}_", dir=str(DEFAULT_ARTIFACT_ROOT))).resolve()


def _apply_llm_overrides(config: dict[str, Any], provider: str | None, model: str | None) -> None:
    if provider is None and model is None:
        return
    llm_cfg = dict(config.get("llm") or {})
    if provider is not None:
        provider_name = str(provider).strip().lower()
        llm_cfg["provider"] = provider_name
        llm_cfg["mode"] = "provider"
        if provider_name == "bigmodel":
            if model is None:
                llm_cfg["model"] = "glm-4.7-flashx"
            llm_cfg["api_key_env"] = "BIGMODEL_API_KEY"
            llm_cfg["base_url"] = "https://open.bigmodel.cn/api/paas/v4"
            llm_cfg.setdefault("thinking_type", "disabled")
        elif provider_name == "openrouter":
            if model is None:
                llm_cfg["model"] = "z-ai/glm-4.5-air:free"
            llm_cfg["api_key_env"] = "OPENROUTER_API_KEY"
            llm_cfg["base_url"] = "https://openrouter.ai/api/v1"
            llm_cfg.pop("thinking_type", None)
    if model is not None:
        llm_cfg["model"] = str(model)
        llm_cfg.setdefault("mode", "provider")
    config["llm"] = llm_cfg


def _prepare_resolved_config(
    case_name: str,
    *,
    config_root: str | Path | None = None,
    graph: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    run_dir: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Return ``(workspace_root, resolved_config_root, resolved_config)``."""

    base_config = deepcopy(load_case_config(case_name, config_root=config_root))
    if graph is not None:
        base_config.setdefault("builder", {})
        base_config["builder"]["graph"] = str(graph).strip()

    _apply_llm_overrides(base_config, provider, model)

    workspace_root = _workspace_root(case_name, run_dir)
    workspace_root.mkdir(parents=True, exist_ok=True)
    resolved_config_root = workspace_root / "config"
    resolved_config_root.mkdir(parents=True, exist_ok=True)

    base_config["save_dir"] = str(workspace_root)
    base_config["state_save_path"] = str(workspace_root / f"{case_name}_built_state.pkl")
    base_config["report_path"] = str(workspace_root / f"{case_name}_final_report.md")

    resolved_path = resolved_config_root / f"{case_name}.yaml"
    resolved_path.write_text(yaml.safe_dump(base_config, sort_keys=False), encoding="utf-8")
    return workspace_root, resolved_config_root, base_config


def run_case_cli(
    case_name: str,
    *,
    config_root: str | Path | None = None,
    graph: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    workspace_root, resolved_config_root, resolved_config = _prepare_resolved_config(
        case_name,
        config_root=config_root,
        graph=graph,
        provider=provider,
        model=model,
        run_dir=run_dir,
    )

    failure_path = workspace_root / "failure.json"
    try:
        payload = run_graph_case(case_name, config_root=resolved_config_root)
    except Exception as exc:
        graph_artifacts = {}
        dag_summary = {}
        state = load_state(str(resolved_config["state_save_path"]))
        if state is not None:
            graph_artifacts = export_dag_artifacts(
                state,
                output_dir=workspace_root / "graphs",
                stem="dag",
                max_nodes=None,
                save_png=True,
                save_json=True,
            )
            dag_summary = summarize_dag_state(state)
            (workspace_root / "dag_summary.json").write_text(
                json.dumps(dag_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        failure_payload = {
            "status": "failed",
            "case_name": case_name,
            "graph": str(resolved_config.get("builder", {}).get("graph", "")),
            "provider": str((resolved_config.get("llm") or {}).get("provider", "")),
            "model": str((resolved_config.get("llm") or {}).get("model", "")),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "graph_artifacts": graph_artifacts,
            "dag_summary": dag_summary,
        }
        failure_path.write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path = workspace_root / "run_manifest.json"
        manifest = {
            "status": "failed",
            "case_name": case_name,
            "config_root": str(resolved_config_root),
            "resolved_config_path": str(resolved_config_root / f"{case_name}.yaml"),
            "run_dir": str(workspace_root),
            "graph": str(resolved_config.get("builder", {}).get("graph", "")),
            "provider": str((resolved_config.get("llm") or {}).get("provider", "")),
            "model": str((resolved_config.get("llm") or {}).get("model", "")),
            "state_save_path": str(resolved_config.get("state_save_path", "")),
            "report_path": str(resolved_config.get("report_path", "")),
            "graph_artifacts": graph_artifacts,
            "dag_summary": dag_summary,
            "failure_path": str(failure_path),
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise

    state = load_state(str(resolved_config["state_save_path"]))
    graph_artifacts = {}
    dag_summary = {}
    if state is not None:
        graph_artifacts = export_dag_artifacts(
            state,
            output_dir=workspace_root / "graphs",
            stem="dag",
            max_nodes=None,
            save_png=True,
            save_json=True,
        )
        dag_summary = summarize_dag_state(state)
        (workspace_root / "dag_summary.json").write_text(
            json.dumps(dag_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    manifest = {
        "case_name": case_name,
        "config_root": str(resolved_config_root),
        "resolved_config_path": str(resolved_config_root / f"{case_name}.yaml"),
        "run_dir": str(workspace_root),
        "graph": str(resolved_config.get("builder", {}).get("graph", "")),
        "provider": str((resolved_config.get("llm") or {}).get("provider", "")),
        "model": str((resolved_config.get("llm") or {}).get("model", "")),
        "state_save_path": str(resolved_config.get("state_save_path", "")),
        "report_path": str(resolved_config.get("report_path", "")),
        "graph_artifacts": graph_artifacts,
        "dag_summary": dag_summary,
        "payload": payload,
    }
    manifest_path = workspace_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    payload = dict(payload)
    payload["resolved_config_path"] = manifest["resolved_config_path"]
    payload["run_manifest_path"] = str(manifest_path)
    payload["run_dir"] = str(workspace_root)
    payload["graph_artifacts"] = graph_artifacts
    payload["dag_summary"] = dag_summary
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Thin CLI wrapper around main.run_case().")
    parser.add_argument("case_name", nargs="?", help="Case name under config/.")
    parser.add_argument("--case", dest="case_flag", help="Case name under config/.")
    parser.add_argument("--config-root", default=None, help="Alternative config root.")
    parser.add_argument("--graph", choices=["builder", "executor", "with_report"], default=None)
    parser.add_argument("--provider", default=None, help="Override llm.provider.")
    parser.add_argument("--model", default=None, help="Override llm.model.")
    parser.add_argument("--run-dir", default=None, help="Directory for resolved config and artifacts.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Alias for --run-dir.",
    )
    args = parser.parse_args(argv)

    if load_dotenv is not None and DOTENV_PATH.exists():
        load_dotenv(DOTENV_PATH)

    case_name = args.case_flag or args.case_name
    if not case_name:
        parser.error("case_name is required via positional argument or --case.")

    run_dir = args.run_dir or args.output_dir
    try:
        payload = run_case_cli(
            case_name,
            config_root=args.config_root,
            graph=args.graph,
            provider=args.provider,
            model=args.model,
            run_dir=run_dir,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
