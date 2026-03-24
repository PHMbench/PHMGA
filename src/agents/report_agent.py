from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from src.configuration import Configuration
from src.dag_artifacts import export_state_artifacts, resolve_artifact_root
from src.model import get_llm
from src.prompts.report_prompt import REPORT_PROMPT
from src.states.phm_states import PHMState


def _summarize_similarity_stats(similarity_stats: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for leaf_id, metric_payload in (similarity_stats or {}).items():
        leaf_summary: Dict[str, Any] = {}
        for metric_name, matrix in (metric_payload or {}).items():
            best = None
            worst = None
            for ref_id, test_map in (matrix or {}).items():
                for test_id, raw_value in (test_map or {}).items():
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        continue
                    candidate = {"ref_id": ref_id, "test_id": test_id, "value": value}
                    if best is None or value > best["value"]:
                        best = candidate
                    if worst is None or value < worst["value"]:
                        worst = candidate
            leaf_summary[metric_name] = {"max": best, "min": worst}
        summary[leaf_id] = leaf_summary
    return summary


def _summarize_ml_results(ml_results: Dict[str, Any]) -> Dict[str, Any]:
    models = {}
    for node_id, payload in (ml_results.get("models") or {}).items():
        metrics = dict((payload or {}).get("metrics") or {})
        models[node_id] = {"metrics": metrics}
    return {
        "models": models,
        "ensemble_metrics": dict(ml_results.get("ensemble_metrics") or {}),
        "weighted_ensemble_metrics": dict(ml_results.get("weighted_ensemble_metrics") or {}),
        "node_level_results": list(ml_results.get("node_level_results") or []),
        "final_selection": dict(ml_results.get("final_selection") or {}),
        "dag_summary": dict(ml_results.get("dag_summary") or {}),
        "protocol_summary": dict(ml_results.get("protocol_summary") or {}),
        "metrics_markdown": str(ml_results.get("metrics_markdown") or ""),
    }


def _summarize_node_level_results(ml_results: Dict[str, Any]) -> Any:
    node_level_results = ml_results.get("node_level_results")
    if node_level_results:
        return node_level_results
    summarized = []
    for node_id, payload in (ml_results.get("models") or {}).items():
        metrics = dict((payload or {}).get("metrics") or {})
        summarized.append(
            {
                "node_id": node_id,
                "metrics": metrics,
                "model_b64_present": bool((payload or {}).get("model_b64")),
            }
        )
    return summarized


def _summarize_final_selection(ml_results: Dict[str, Any]) -> Dict[str, Any]:
    final_selection = ml_results.get("final_selection")
    if isinstance(final_selection, dict) and final_selection:
        return final_selection
    weighted = dict(ml_results.get("weighted_ensemble_metrics") or ml_results.get("ensemble_metrics") or {})
    if not weighted:
        return {}
    return {
        "weighted_ensemble": weighted,
    }


def report_agent(
    *,
    instruction: str,
    dag_overview: Dict[str, Any],
    similarity_stats: Dict[str, Any],
    ml_results: Dict[str, Any],
    node_level_results: Optional[Any] = None,
    final_selection: Optional[Dict[str, Any]] = None,
    issues_summary: Optional[str] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate a final markdown report via LLM."""

    llm = get_llm(runtime_config or {"llm": Configuration.from_runnable_config(None).model_dump()})
    node_level_results_payload = node_level_results if node_level_results is not None else _summarize_node_level_results(ml_results)
    final_selection_payload = final_selection if final_selection is not None else _summarize_final_selection(ml_results)
    llm_cfg = dict((runtime_config or {}).get("llm") or {})
    runtime_summary = {
        "provider": str(llm_cfg.get("provider", "")),
        "model": str(llm_cfg.get("model", "")),
    }
    prompt = REPORT_PROMPT.format(
        instruction=instruction,
        dag_overview=json.dumps(dag_overview, ensure_ascii=False),
        similarity_stats=json.dumps(_summarize_similarity_stats(similarity_stats), ensure_ascii=False),
        ml_results=json.dumps(_summarize_ml_results(ml_results), ensure_ascii=False),
        node_level_results=json.dumps(node_level_results_payload, ensure_ascii=False),
        final_selection=json.dumps(final_selection_payload, ensure_ascii=False),
        runtime_summary=json.dumps(runtime_summary, ensure_ascii=False),
        issues_summary=issues_summary or "",
    )
    if getattr(llm, "mode", "") == "offline_stub":
        return {"report_markdown": prompt}
    report_markdown = str(llm.generate_text(prompt) or "").strip()
    if report_markdown:
        return {"report_markdown": report_markdown}

    retry_prompt = (
        "请只输出 Markdown 报告，不要留空。\n\n"
        + prompt
    )
    return {"report_markdown": str(llm.generate_text(retry_prompt) or "").strip()}


def report_agent_node(state: PHMState) -> Dict[str, str]:
    """Adapter using :class:`PHMState` for the outer graph."""
    artifact_root = resolve_artifact_root(state.runtime_config or {}, state.case_name)
    graph_artifacts = export_state_artifacts(
        state,
        output_dir=artifact_root / "graphs",
        stem="final_dag",
        max_nodes=None,
        save_png=True,
        save_json=True,
    )
    if graph_artifacts["warnings"]:
        state.dag_state.error_log.extend(graph_artifacts["warnings"])
    dag_overview = json.loads(state.tracker().export_json(max_nodes=None))
    
    # MODIFIED: Extract similarity stats from the `sim` attribute of leaf nodes
    similarity_stats = {}
    for leaf_id in state.dag_state.leaves:
        node = state.dag_state.nodes.get(leaf_id)
        if node and hasattr(node, "sim") and node.sim:
            similarity_stats[leaf_id] = node.sim
            
    ml_results = getattr(state, "ml_results", {}) or {}
    issues_summary = "\n".join(state.dag_state.error_log) or None
    # calculate the payload for each section
    print(f"Generating final report with {len(state.dag_state.nodes)} nodes, "
          f"{len(state.dag_state.leaves)} leaves, "
          f"{len(similarity_stats)} similarity stats, "
          f"{len((ml_results.get('models') or {}))} ML models, "
          f"{len((ml_results.get('node_level_results') or []))} node results, "
          f"issues: {len(state.dag_state.error_log)}")
    out = report_agent(
        instruction=state.user_instruction,
        dag_overview=dag_overview,
        similarity_stats=similarity_stats,
        ml_results=ml_results,
        node_level_results=ml_results.get("node_level_results"),
        final_selection=ml_results.get("final_selection"),
        issues_summary=issues_summary,
        runtime_config=state.runtime_config or None,
    )
    return {"final_report": out["report_markdown"]}


if __name__ == "__main__":
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
