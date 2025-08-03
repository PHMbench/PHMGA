import os
from langchain_community.chat_models import FakeListChatModel

from src.agents.report_agent import report_agent


def test_report_agent_generates_markdown():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(responses=["# Report\n\n" + "x" * 600])

    dag_overview = {"nodes": ["n1"], "graph_path": "dag.png"}
    similarity_stats = {"new_nodes": ["sim_cosine_ch1"]}
    ml_results = {
        "models": {},
        "ensemble_metrics": {"accuracy": 0.9},
        "metrics_markdown": "| model | accuracy |\n|---|---|\n| m | 0.9 |",
    }
    out = report_agent(
        instruction="diag", dag_overview=dag_overview, similarity_stats=similarity_stats, ml_results=ml_results
    )
    assert list(out.keys()) == ["report_markdown"]
    assert len(out["report_markdown"]) > 500
