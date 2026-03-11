import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from src.cases.base_runner import run_registered_case
from src.utils.logging_setup import (
    clear_current_logger,
    init_run_logger,
    log_event,
    set_current_logger,
)
from src.utils.preflight import run_preflight_from_config_path


def _run_preflight(config_path: str) -> int:
    report = run_preflight_from_config_path(config_path)
    print(f"Preflight config: {config_path}")
    print(f"OK: {report.get('ok')}")
    if report.get("errors"):
        print("Errors:")
        for item in report["errors"]:
            print(f"- {item}")
    if report.get("warnings"):
        print("Warnings:")
        for item in report["warnings"]:
            print(f"- {item}")
    return 0 if report.get("ok") else 2


def _resolve_compose_target(args: argparse.Namespace) -> tuple[Path, str]:
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return config_path.parent, config_path.stem
    config_dir = Path(args.config_dir or "config").resolve()
    config_name = str(args.config_name or "config").strip() or "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    target_path = config_dir / f"{config_name}.yaml"
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")
    return config_dir, config_name


def run_main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if argv and argv[0] == "preflight":
        preflight_parser = argparse.ArgumentParser(description="Run PHM preflight checks.")
        preflight_parser.add_argument("--config", type=str, required=True, help="Path to case configuration yaml.")
        preflight_args = preflight_parser.parse_args(argv[1:])
        return _run_preflight(preflight_args.config)

    parser = argparse.ArgumentParser(description="Run PHM analysis cases through Hydra compose + case registry.")
    parser.add_argument(
        "legacy_case_name",
        nargs="?",
        default=None,
        help="Legacy case runner name. Mapped to cases.selected when provided.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file composed via Hydra from its directory.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Hydra config directory (default: config).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name without suffix (default: config).",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Explicit case runner override. Preferred over the legacy positional name.",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Override graphs.selected for this run.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be repeated.",
    )
    args, unknown = parser.parse_known_args(argv)

    hydra_overrides = list(args.override or []) + list(unknown or [])
    selected_case = str(args.case or args.legacy_case_name or "").strip() or None
    if args.graph:
        hydra_overrides.append(f"graphs.selected={args.graph}")

    try:
        config_dir, config_name = _resolve_compose_target(args)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    resolved_config_path = config_dir / f"{config_name}.yaml"

    run_label = selected_case or config_name
    run_id = f"main-{int(time.time())}"
    save_dir = os.getenv("PHM_SAVE_DIR") or os.path.join(os.getcwd(), "save")
    bundle = init_run_logger(case_name=run_label, save_dir=save_dir, run_id=run_id)
    set_current_logger(bundle)

    try:
        log_event(
            bundle,
            level="INFO",
            event="main.start",
            phase="main",
            message="Starting case runner via Hydra compose.",
            payload={
                "config_dir": str(config_dir),
                "config_name": config_name,
                "config_path": str(resolved_config_path),
                "hydra_overrides": hydra_overrides,
                "case_name": selected_case,
            },
        )
        runtime = run_registered_case(
            config_dir=config_dir,
            config_name=config_name,
            hydra_overrides=hydra_overrides,
            case_name=selected_case,
        )
        log_event(
            bundle,
            level="INFO",
            event="main.success",
            phase="main",
            message="Case finished.",
            payload=runtime,
        )
        return 0
    except Exception as e:
        log_event(
            bundle,
            level="ERROR",
            event="main.fail",
            phase="main",
            message=f"Case failed: {e}",
            payload={
                "config_dir": str(config_dir),
                "config_name": config_name,
                "hydra_overrides": hydra_overrides,
                "case_name": selected_case,
            },
        )
        print(f"An error occurred while running case '{run_label}': {e}")
        return 2
    finally:
        clear_current_logger()


def main() -> int:
    return run_main(sys.argv[1:])

if __name__ == "__main__":
    sys.exit(main())
