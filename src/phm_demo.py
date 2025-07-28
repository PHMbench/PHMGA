"""Minimal demonstration pipeline for PHM operators."""

from __future__ import annotations

from typing import TypedDict, List, Dict

import numpy as np
from langgraph.graph import StateGraph, END

from src.tools.schemas import PHMOperator, SimilarityOperator
from src.tools.factory import create_operator

class DemoState(TypedDict):
    """Shared state for the demo graph."""

    plan: List[PHMOperator]
    reference_signals: Dict[str, np.ndarray]
    test_signal: np.ndarray
    result: str
    output: str | None


def generate_signals() -> DemoState:
    """Create a starting state with random signals and a simple plan."""

    rng = np.random.default_rng(0)

    ref_signals = {
        f"class_{i}": rng.normal(size=(1, 128, 1)) for i in range(4)
    }
    test_signal = rng.normal(size=(1, 128, 1))

    # The plan contains two operators executed sequentially.
    plan_dicts = [
        {"op_name": "patch", "patch_size": 32, "stride": 16},
        {"op_name": "mean"},
    ]
    plan = [create_operator(d) for d in plan_dicts]

    return {
        "plan": plan,
        "reference_signals": ref_signals,
        "test_signal": test_signal,
        "result": "",
        "output": None,
    }


def planner(state: DemoState) -> DemoState:
    """Return the state unchanged (placeholder for LLM planner)."""

    return state


def tool_executor(state: DemoState) -> DemoState:
    """Execute the plan on test and reference signals."""

    data = state["test_signal"]
    for op in state["plan"]:
        data = op.execute(data)
    state["test_signal"] = data

    ref_processed: Dict[str, np.ndarray] = {}
    for label, sig in state["reference_signals"].items():
        ref = sig
        for op in state["plan"]:
            ref = op.execute(ref)
        ref_processed[label] = ref

    state["reference_signals"] = ref_processed
    return state


def decide(state: DemoState) -> DemoState:
    """Select the reference signal most similar to the test signal."""

    test_feat = state["test_signal"]
    best_label = ""
    best_score = 0.0

    for label, feat in state["reference_signals"].items():
        sim_op = SimilarityOperator()
        if sim_op.execute(test_feat, ref=feat):
            dot = np.vdot(test_feat, feat)
            norm = np.linalg.norm(test_feat) * np.linalg.norm(feat)
            score = float(abs(dot) / norm)
            if score > best_score:
                best_score = score
                best_label = label

    state["result"] = best_label or "unknown"
    return state


def reporter(state: DemoState) -> DemoState:
    """Format the final result for display."""

    state["output"] = f"Predicted label: {state['result']}"
    return state


# Build graph
builder = StateGraph(DemoState)
builder.add_node("planner", planner)
builder.add_node("execute", tool_executor)
builder.add_node("decide", decide)
builder.add_node("report", reporter)

builder.set_entry_point("planner")
builder.add_edge("planner", "execute")
builder.add_edge("execute", "decide")
builder.add_edge("decide", "report")
builder.add_edge("report", END)

demo_graph = builder.compile()


def main() -> None:
    """Run the demo graph and print the classification result."""

    state = generate_signals()
    result = demo_graph.invoke(state)
    print(result["output"])


if __name__ == "__main__":
    main()
