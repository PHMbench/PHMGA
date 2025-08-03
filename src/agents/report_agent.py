from __future__ import annotations

import json
from typing import Any, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from src.configuration import Configuration
from src.model import get_llm
from src.prompts.report_prompt import REPORT_PROMPT
from phm_core import PHMState


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
    return {"report_markdown": resp.content}


def report_agent_node(state: PHMState) -> Dict[str, str]:
    """Adapter using :class:`PHMState` for the outer graph."""
    try:  # generate final DAG image
        state.tracker().write_png("final_dag.png")
    except Exception:
        pass
    dag_overview = {
        "n_nodes": len(state.dag_state.nodes),
        "leaves": state.dag_state.leaves,
        "graph_path": state.dag_state.graph_path or "",
    }
    similarity_stats = {"new_nodes": [nid for nid in state.dag_state.nodes if nid.startswith("sim_")]}
    ml_results = getattr(state, "ml_results", {}) or {}
    issues_summary = "\n".join(state.dag_state.error_log) or None
    out = report_agent(
        instruction=state.user_instruction,
        dag_overview=dag_overview,
        similarity_stats=similarity_stats,
        ml_results=ml_results,
        issues_summary=issues_summary,
    )
    return {"final_report": out["report_markdown"]}


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
    ml_results = {
        "models": {},
        "ensemble_metrics": {"accuracy": 0.95, "f1": 0.94},
        "metrics_markdown": "| model | accuracy | f1 |\n|---|---|---|\n| m | 0.9 | 0.8 |",
    }
    print({"before": state.model_dump(exclude={"reference_signal", "test_signal"})})
    out = report_agent(
        instruction=instruction,
        dag_overview={"n_nodes": len(state.dag_state.nodes), "leaves": state.dag_state.leaves, "graph_path": state.dag_state.graph_path or ""},
        similarity_stats={"new_nodes": []},
        ml_results=ml_results,
    )
    print({"after": len(out["report_markdown"])})
