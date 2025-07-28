from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import logging

from .configuration import Config
from .state import SignalData, PHMState
from .graph import build_graph

logging.basicConfig(level=logging.INFO)


def load_signal(path: Path) -> SignalData:
    arr = np.load(path)
    return SignalData(data=arr.tolist(), sampling_rate=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PHM pipeline")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--reference_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--use_patch", action="store_true")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_loops", type=int, default=3)
    parser.add_argument("--output", default="report.md")
    args = parser.parse_args()

    config = Config(
        use_patch=args.use_patch,
        patch_size=args.patch_size,
        max_loops=args.max_loops,
        signal_processing_methods=["identity"],
        feature_methods=["mean"],
        similarity_method="l2",
        decision_model="simple",
    )

    reference_signal = load_signal(Path(args.reference_path))
    test_signal = load_signal(Path(args.test_path))

    state: PHMState = {
        "user_instruction": args.instruction,
        "reference_signal": reference_signal,
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

    graph = build_graph(config)
    final_state = graph.invoke(state, {"configurable": {"thread_id": "run"}})

    with open(args.output, "w") as f:
        f.write(final_state["final_report"])
    print("Report written to", args.output)


if __name__ == "__main__":
    main()
