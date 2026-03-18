"""Unified Hydra entrypoint for the paper-oriented PHMGA workflow."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

from scripts.preflight import run_preflight
from scripts.run_case import run_case
from src.config import load_runtime_config

# Load .env file before importing any modules that might need env vars
load_dotenv()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose one runtime config and dispatch to preflight or run_case."""

    runtime_config = load_runtime_config(cfg)
    action = str(runtime_config.get("runtime", {}).get("action", "run_case"))
    if action == "preflight":
        payload = run_preflight(runtime_config)
    elif action == "run_case":
        payload = run_case(runtime_config, output_dir=runtime_config["runtime"].get("output_dir"))
    else:
        raise ValueError(f"Unsupported runtime.action: {action}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
