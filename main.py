"""Unified CLI for the simplified thesis_2026 runtime."""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from src.config import load_runtime_config
from src.runtime import run_experiment, run_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a thesis_2026 preset.")
    parser.add_argument("run_name", help="Run preset under config/runs/*.yaml or a legacy case alias.")
    parser.add_argument("--action", choices=("run", "preflight"), default="run")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values with dotted.key=value.",
    )
    args = parser.parse_args()

    load_dotenv()
    runtime_config = load_runtime_config(
        args.run_name,
        action=args.action,
        output_dir=args.output_dir,
        overrides=args.set,
    )
    if args.action == "preflight":
        payload = run_preflight(runtime_config)
    else:
        payload = run_experiment(runtime_config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
