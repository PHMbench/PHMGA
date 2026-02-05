from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.report_agent import report_agent_node
from src.utils import load_state


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate final report from saved PHMState (and optional ML results).")
    ap.add_argument("--state", required=True, help="Path to saved PHMState pickle.")
    ap.add_argument("--ml-results", default=None, help="Optional ml_results.pkl to attach to state.")
    ap.add_argument("--out", default="final_report.md", help="Output markdown path.")
    args = ap.parse_args()

    state = load_state(args.state)
    if state is None:
        raise SystemExit(f"Failed to load state: {args.state}")

    if args.ml_results:
        with open(args.ml_results, "rb") as f:
            ml = pickle.load(f)
        # Keep only report-relevant fields to avoid huge dumps.
        state.ml_results = {
            "ensemble_metrics": ml.get("ensemble_metrics", {}),
            "metrics_markdown": ml.get("metrics_markdown", ""),
            "tspn": ml.get("tspn", {}),
        }

    out = report_agent_node(state)
    report = out.get("final_report", "")
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"Wrote report -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    main()
