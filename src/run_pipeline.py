"""Command line interface to run the PHM pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .configuration import Config
from .graph import build_graph
from .state import PHMState, SignalData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_signal(path: Path) -> SignalData:
    arr = np.load(path)
    return SignalData(data=arr.tolist(), sampling_rate=100)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--tests_path", required=True)
    parser.add_argument("--use_patch", action="store_true")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_loops", type=int, default=3)
    args = parser.parse_args(argv)

    config = Config(
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        signal_processing_methods=["identity"],
        feature_methods=["mean"],
        similarity_method="euclidean",
        decision_model="threshold",
        max_loops=args.max_loops,
    )
    graph = build_graph(config)

    ref_signal = load_signal(Path(args.reference_path))
    test_signal = load_signal(Path(args.tests_path))

    state: PHMState = {
        "user_instruction": args.instruction,
        "reference_signal": ref_signal,
        "test_signal": test_signal,
        "plan": {},
        "reflection_history": [],
        "is_sufficient": False,
        "iteration_count": 0,
        "processed_signals": [],
        "extracted_features": [],
        "analysis_results": [],
        "final_decision": "",
        "final_report": "",
    }

    result = graph.invoke(state, config={"configurable": {"thread_id": "0"}})
    with open("report.md", "w") as f:
        f.write(result["final_report"])
    print("Decision:", result["final_decision"])


if __name__ == "__main__":
    main()
