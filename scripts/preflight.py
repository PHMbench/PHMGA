"""Preflight checks for config, protocol, operators, and graph paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.operators import get_operator_catalog


def run_preflight(config_input: Union[str, Path, Dict[str, Any]]) -> dict:
    """Validate one resolved config against the shared runtime contract."""
    runtime_config = load_runtime_config(config_input)
    protocol = build_protocol_from_config(runtime_config)
    catalog = get_operator_catalog()
    return {
        "status": "ok",
        "config_name": runtime_config["runtime"]["config_name"],
        "dataset_name": protocol.dataset_name,
        "graph_path": runtime_config["experiment"]["graph_path"],
        "source_mode": protocol.source_mode,
        "sample_count": len(protocol.samples),
        "splits": {
            "train": len(protocol.splits.train_ids),
            "val": len(protocol.splits.val_ids),
            "test": len(protocol.splits.test_ids),
        },
        "window": protocol.window.model_dump(),
        "selected_channels": protocol.selected_channels,
        "operators": [spec.op_uid for spec in catalog.specs()],
    }


def main() -> None:
    """CLI entrypoint for repository preflight checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
