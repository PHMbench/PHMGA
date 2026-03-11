import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_OPENROUTER_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(
    not _enabled(),
    reason="Set PHM_ENABLE_OPENROUTER_TESTS=1 to enable online OpenRouter smoke tests.",
)
def test_openrouter_invoke_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FAKE_LLM", raising=False)
    os.environ["LLM_PROVIDER"] = "openrouter"
    os.environ.setdefault("QUERY_GENERATOR_MODEL", "openai/gpt-4o-mini")

    if not os.getenv("OPENROUTER_BASE_URL") or not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("Missing OPENROUTER_BASE_URL/OPENROUTER_API_KEY.")

    from src.model import get_llm

    try:
        llm = get_llm(temperature=0)
    except ImportError as exc:
        if "langchain_openai" in str(exc):
            pytest.skip(str(exc))
        raise

    resp = llm.invoke("Reply exactly OK")
    content = str(getattr(resp, "content", resp))
    assert content.strip()
