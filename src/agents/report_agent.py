from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.report_prompt import REPORT_PROMPT
from src.states.phm_states import PHMState


def report_agent(
    *,
    instruction: str,
    dag_overview: Dict[str, Any],
    similarity_stats: Dict[str, Any],
    ml_results: Dict[str, Any],
    issues_summary: Optional[str] = None,
) -> Dict[str, str]:
    """Generate a final markdown report via LLM."""

    llm = get_llm(Configuration.from_runnable_config(None))
    prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
    chain = prompt | llm
    resp = chain.invoke(
        {
            "instruction": instruction,
            "dag_overview": json.dumps(dag_overview, ensure_ascii=False),
            "similarity_stats": json.dumps(similarity_stats, ensure_ascii=False),
            "ml_results": json.dumps(ml_results, ensure_ascii=False),
            "issues_summary": issues_summary or "",
        }
    )
    if os.getenv("PHM_DEBUG_REPORT", "").strip().lower() in {"1", "true", "yes", "y"}:
        print("\n--- Report Agent LLM Response ---")
        print(resp.content)
        print("--------------------------------\n")
    return {"report_markdown": resp.content}


def _template_report(
    *,
    instruction: str,
    dag_overview: Dict[str, Any],
    similarity_stats: Dict[str, Any],
    ml_results: Dict[str, Any],
    issues_summary: Optional[str],
) -> str:
    tspn = (ml_results or {}).get("tspn") or {}
    metrics = tspn.get("metrics") or {}
    val = metrics.get("val") or {}
    best = metrics.get("best") or {}
    lines = []
    lines.append("# PHMGA Diagnostic Report (Template)")
    lines.append("")
    lines.append("## Conclusion")
    if val:
        lines.append(f"- Val acc: {val.get('val_acc')}")
        lines.append(f"- Val macro_f1: {val.get('val_macro_f1')}")
    if "test_acc" in metrics:
        lines.append(f"- Test acc: {metrics.get('test_acc')}")
        lines.append(f"- Test macro_f1: {metrics.get('test_macro_f1')}")
    if best:
        lines.append(f"- Best epoch: {best.get('epoch')}")
    lines.append("")
    lines.append("## Evidence")
    if isinstance(dag_overview, dict):
        nodes = dag_overview.get("nodes")
        graph = dag_overview.get("graph")
        if isinstance(nodes, list):
            n_nodes = len(nodes)
        elif isinstance(graph, list):
            n_nodes = len(graph)
        else:
            n_nodes = 0
    else:
        n_nodes = "n/a"
    lines.append(f"- DAG nodes: {n_nodes}")
    lines.append(f"- Similarity stats: {len(similarity_stats) if isinstance(similarity_stats, dict) else 'n/a'}")
    if tspn.get("artifacts_dir"):
        lines.append(f"- Artifacts: `{tspn.get('artifacts_dir')}`")
    if issues_summary:
        lines.append("")
        lines.append("## Issues")
        lines.append(issues_summary)
    lines.append("")
    lines.append("## Original Instruction")
    lines.append(instruction.strip())
    return "\n".join(lines).strip() + "\n"


def report_agent_node(state: PHMState) -> Dict[str, str]:
    """Adapter using :class:`PHMState` for the outer graph."""
    try:  # generate final DAG image
        base_save_dir = (
            getattr(state, "save_dir", None)
            or os.environ.get("PHM_SAVE_DIR")
            or os.environ.get("PHM_DATA_DIR")
            or os.path.join(os.getcwd(), "save")
        )
        case_name = getattr(state, "case_name", "") or "case"
        save_path = os.path.join(base_save_dir, case_name, "final_dag.png")
        state.tracker().write_png(save_path)
    except Exception:
        pass
    dag_overview = json.loads(state.tracker().export_json())
    
    # Extract similarity stats from the `sim` attribute of leaf nodes.
    similarity_stats = {}
    for leaf_id in state.dag_state.leaves:
        node = state.dag_state.nodes.get(leaf_id)
        if node and hasattr(node, "sim") and node.sim:
            similarity_stats[leaf_id] = node.sim

    ml_results = getattr(state, "ml_results", {}) or {}
    issues_summary = "\n".join(state.dag_state.error_log) or None
    if os.getenv("PHM_DEBUG_REPORT", "").strip().lower() in {"1", "true", "yes", "y"}:
        print(
            f"Generating final report with {len(state.dag_state.nodes)} nodes, "
            f"{len(state.dag_state.leaves)} leaves, "
            f"{len(similarity_stats)} similarity stats, "
            f"issues: {len(state.dag_state.error_log)}"
        )
    mode = os.getenv("PHM_REPORT_MODE", "auto").strip().lower()
    fake_llm = os.getenv("FAKE_LLM", "").strip().lower() in {"1", "true", "yes", "y"}
    if mode == "template" or (mode == "auto" and fake_llm):
        return {
            "final_report": _template_report(
                instruction=state.user_instruction,
                dag_overview=dag_overview,
                similarity_stats=similarity_stats,
                ml_results=ml_results,
                issues_summary=issues_summary,
            )
        }

    try:
        out = report_agent(
            instruction=state.user_instruction,
            dag_overview=dag_overview,
            similarity_stats=similarity_stats,
            ml_results=ml_results,
            issues_summary=issues_summary,
        )
        return {"final_report": out["report_markdown"]}
    except Exception:
        # Fallback to template for robustness.
        return {
            "final_report": _template_report(
                instruction=state.user_instruction,
                dag_overview=dag_overview,
                similarity_stats=similarity_stats,
                ml_results=ml_results,
                issues_summary=issues_summary,
            )
        }


if __name__ == "__main__":
    import os
    import sys
    from langchain_community.chat_models import FakeListChatModel

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    os.environ["FAKE_LLM"] = "true"
    from src import model
    from phm_core import PHMState, DAGState, InputData

    model._FAKE_LLM = FakeListChatModel(responses=["# Demo Report\nThis is a demo report." * 20])

    instruction = "轴承故障诊断"
    ch1 = InputData(node_id="ch1", data={}, parents=[], shape=(0,))
    ch2 = InputData(node_id="ch2", data={}, parents=[], shape=(0,))
    dag = DAGState(user_instruction=instruction, channels=["ch1", "ch2"], nodes={"ch1": ch1, "ch2": ch2}, leaves=["ch1", "ch2"], graph_path="dag.png")
    state = PHMState(user_instruction=instruction, reference_signal=ch1, test_signal=ch2, dag_state=dag)
    
    # Add mock similarity data to a leaf node for testing
    ch1.sim = {"euclidean": {"id1": {"id4": 0.5}}}

    ml_results = {
        "models": {},
        "ensemble_metrics": {"accuracy": 0.95, "f1": 0.94},
        "metrics_markdown": "| model | accuracy | f1 |\n|---|---|---|\n| m | 0.9 | 0.8 |",
    }
    print({"before": state.model_dump(exclude={"reference_signal", "test_signal"})})
    
    # Use the node adapter for a more realistic test
    out = report_agent_node(state)

    print({"after": len(out["final_report"])})
    assert "This is a demo report" in out["final_report"]
    print("✅ Report Agent test passed!")
