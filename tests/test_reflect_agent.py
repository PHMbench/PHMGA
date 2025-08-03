import os
from langchain_community.chat_models import FakeListChatModel

from src.agents.reflect_agent import reflect_agent


def test_reflect_agent_proceed():
    os.environ["FAKE_LLM"] = "true"
    from src import model
    model._FAKE_LLM = FakeListChatModel(
        responses=['{"decision": "proceed", "reason": "ok"}']
    )
    blueprint = {"nodes": ["a"], "edges": []}
    out = reflect_agent(instruction="goal", stage="POST_PLAN", dag_blueprint=blueprint)
    assert out["decision"] == "proceed"
    assert out["reason"] == "ok"
