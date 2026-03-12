from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_runtime_config
from src.data import build_protocol_from_config
from src.operators import get_operator_catalog


def run_preflight(config_path: str) -> dict:
    catalog = get_operator_catalog()
    datasets = {}
    for dataset_name in ("RM101", "Ottawa"):
        runtime_config = load_runtime_config(config_path, dataset_name=dataset_name)
        protocol = build_protocol_from_config(runtime_config)
        datasets[dataset_name] = {
            "samples": len(protocol.samples),
            "train": len(protocol.splits.train_ids),
            "val": len(protocol.splits.val_ids),
            "test": len(protocol.splits.test_ids),
            "window_size": protocol.window.window_size,
        }
    return {
        "status": "ok",
        "datasets": datasets,
        "graph_paths": ["dag_only", "ml", "torch"],
        "operators": [spec.op_uid for spec in catalog.specs()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
