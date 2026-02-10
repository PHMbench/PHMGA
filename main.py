import argparse
import importlib
import os
import sys
import time

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


def main() -> int:
    """
    Main entry point for running PHM analysis cases.
    Dynamically loads and runs a case module based on command-line arguments.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "preflight":
        preflight_parser = argparse.ArgumentParser(description="Run PHM preflight checks.")
        preflight_parser.add_argument("--config", type=str, required=True, help="Path to case configuration yaml.")
        preflight_args = preflight_parser.parse_args(sys.argv[2:])
        return _run_preflight(preflight_args.config)

    parser = argparse.ArgumentParser(description="Run PHM analysis cases.")
    parser.add_argument(
        "case_name",
        type=str,
        help="The name of the case to run (e.g., 'case1').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file. Defaults to 'config/<case_name>.yaml'.",
    )
    args = parser.parse_args()

    case_name = args.case_name
    config_path = args.config or f"config/{case_name}.yaml"
    run_id = f"main-{int(time.time())}"
    save_dir = os.getenv("PHM_SAVE_DIR") or os.path.join(os.getcwd(), "save")
    bundle = init_run_logger(case_name=case_name, save_dir=save_dir, run_id=run_id)
    set_current_logger(bundle)

    if not os.path.exists(config_path):
        log_event(
            bundle,
            level="ERROR",
            event="main.config_missing",
            phase="main",
            message=f"Configuration file not found: {config_path}",
            payload={"config_path": config_path},
        )
        return 2

    try:
        log_event(
            bundle,
            level="INFO",
            event="main.start",
            phase="main",
            message="Starting case runner.",
            payload={"case_name": case_name, "config_path": config_path},
        )
        # Dynamically import the case module
        case_module = importlib.import_module(f"src.cases.{case_name}")
        
        # Run the case
        case_module.run_case(config_path)
        log_event(
            bundle,
            level="INFO",
            event="main.success",
            phase="main",
            message="Case finished.",
            payload={"case_name": case_name},
        )
        return 0
        
    except ImportError:
        log_event(
            bundle,
            level="ERROR",
            event="main.import_error",
            phase="main",
            message=f"Case '{case_name}' not found.",
            payload={"case_name": case_name},
        )
        return 2
    except Exception as e:
        log_event(
            bundle,
            level="ERROR",
            event="main.fail",
            phase="main",
            message=f"Case failed: {e}",
            payload={"case_name": case_name},
        )
        print(f"An error occurred while running case '{case_name}': {e}")
        return 2
    finally:
        clear_current_logger()

if __name__ == "__main__":
    sys.exit(main())
