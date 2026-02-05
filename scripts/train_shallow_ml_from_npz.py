from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.shallow_ml_agent import shallow_ml_agent


def _load_all_datasets(folder: Path) -> Dict[str, Dict[str, np.ndarray]]:
    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    for p in sorted(folder.glob("*.npz")):
        node_id = p.name.replace("_dataset.npz", "")
        data = np.load(p, allow_pickle=False)
        datasets[node_id] = {
            "X_train": data.get("X_train", np.array([])),
            "y_train": data.get("y_train", np.array([])),
            "X_test": data.get("X_test", np.array([])),
            "y_test": data.get("y_test", np.array([])),
        }
    return datasets


def main() -> None:
    ap = argparse.ArgumentParser(description="Train shallow ML from exported node datasets (*.npz).")
    ap.add_argument("--dataset-dir", required=True, help="Folder containing *_dataset.npz files.")
    ap.add_argument("--out-pkl", default="ml_results.pkl", help="Where to save pickle results.")
    ap.add_argument("--out-md", default="shallow_ml_results.md", help="Where to save markdown table.")
    ap.add_argument("--algorithm", default="RandomForest", help="Estimator name (see shallow_ml_agent).")
    ap.add_argument("--ensemble-method", default="hard_voting", help="hard_voting|soft_voting.")
    ap.add_argument("--cv-folds", type=int, default=5, help="CV folds.")
    args = ap.parse_args()

    folder = Path(args.dataset_dir)
    datasets = _load_all_datasets(folder)
    print(f"Loaded {len(datasets)} node datasets from {folder.resolve()}")

    results = shallow_ml_agent(
        datasets=datasets,
        algorithm=str(args.algorithm),
        ensemble_method=str(args.ensemble_method),
        cv_folds=int(args.cv_folds),
    )

    Path(args.out_md).write_text(results.get("metrics_markdown", ""), encoding="utf-8")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved markdown -> {Path(args.out_md).resolve()}")
    print(f"Saved pickle   -> {Path(args.out_pkl).resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    main()
