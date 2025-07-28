"""Demonstration pipeline for PHM operators and LangGraph."""

from __future__ import annotations

from typing import Dict, List, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from src.tools.factory import create_operator
from src.tools.schemas import PHMOperator, SimilarityOperator

class DemoState(TypedDict):
    """State used by the demonstration graph."""

    plan: List[PHMOperator]
    reference_signals: Dict[str, np.ndarray]
    test_signal: np.ndarray
    result: str
    output: str | None


def generate_signals() -> DemoState:
    """Generate random signals and a simple processing plan."""

    rng = np.random.default_rng(0)
    ref_signals = {f"class_{i}": rng.normal(size=(1, 128, 1)) for i in range(4)}
    test_signal = rng.normal(size=(1, 128, 1))

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
    """Placeholder planner simply returns the provided state."""

    return state


def tool_executor(state: DemoState) -> DemoState:
    """Run each operator on the test and reference signals."""

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
    """Assign the label with highest similarity to the test signal."""

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
    """Format the classification result."""

    state["output"] = f"Predicted label: {state['result']}"
    return state


def build_demo_graph() -> StateGraph:
    """Return a compiled demonstration graph."""

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

    return builder.compile()


__all__ = ["DemoState", "build_demo_graph", "generate_signals"]


if __name__ == "__main__":
    graph = build_demo_graph()
    state = generate_signals()
    result = graph.invoke(state)
    print(result["output"])
