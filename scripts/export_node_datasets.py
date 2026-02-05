from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.dataset_preparer_agent import dataset_preparer_agent
from src.utils import load_state


def _save_datasets(datasets: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for node_id, data in datasets.items():
        file = out_dir / f"{node_id}_dataset.npz"
        np.savez(
            file,
            X_train=data.get("X_train", np.array([])),
            y_train=data.get("y_train", np.array([])),
            X_test=data.get("X_test", np.array([])),
            y_test=data.get("y_test", np.array([])),
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export node datasets from a saved PHMState (.pkl).")
    ap.add_argument("--state", required=True, help="Path to saved PHMState pickle.")
    ap.add_argument("--out-dir", default="generated_datasets", help="Output directory for *.npz datasets.")
    ap.add_argument("--stage", default="processed", help="Which DAG stage to export (default: processed).")
    ap.add_argument("--flatten", action="store_true", help="Flatten legacy .npy feature files if needed.")
    ap.add_argument(
        "--allow-test-labels-for-reporting",
        action="store_true",
        help="Allow using labels_tst to build y_test (reporting-only).",
    )
    args = ap.parse_args()

    state = load_state(args.state)
    if state is None:
        raise SystemExit(f"Failed to load state: {args.state}")

    state.allow_test_labels_for_reporting = bool(args.allow_test_labels_for_reporting)

    out = dataset_preparer_agent(state, config={"stage": args.stage, "flatten": bool(args.flatten)})
    datasets = out.get("datasets", {}) or {}

    out_dir = Path(args.out_dir)
    _save_datasets(datasets, out_dir)
    print(f"Exported {len(datasets)} datasets to {out_dir.resolve()}")


if __name__ == "__main__":
    # Allow running as: python scripts/export_node_datasets.py --state save/.../state.pkl
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    main()
