import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_GEMINI_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_GEMINI_TESTS=1 to enable online Gemini gateway smoke tests.")
def test_gemini_gateway_invoke_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FAKE_LLM", raising=False)
    os.environ["LLM_PROVIDER"] = "openai_compatible"
    os.environ["QUERY_GENERATOR_MODEL"] = "gemini-2.5-flash"

    if not os.getenv("OPENAI_BASE_URL"):
        base = os.getenv("GEMINI_BASE_URL") or os.getenv("GEMINI_BASE")
        if base:
            os.environ["OPENAI_BASE_URL"] = base
    if not os.getenv("OPENAI_API_KEY"):
        key = os.getenv("GEMINI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key

    if not os.getenv("OPENAI_BASE_URL") or not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Missing gateway env vars: OPENAI_BASE_URL/OPENAI_API_KEY (or GEMINI_BASE/GEMINI_API_KEY).")

    from src.configuration import Configuration
    from src.model import get_llm

    try:
        llm = get_llm(Configuration.from_runnable_config(None), temperature=0)
    except ImportError as e:
        if "langchain_openai" in str(e):
            pytest.skip(str(e))
        raise
    except ValueError as e:
        msg = str(e).lower()
        if "missing" in msg and ("api key" in msg or "base_url" in msg or "base url" in msg):
            pytest.skip(str(e))
        raise

    resp = llm.invoke("Reply exactly OK")
    content = str(getattr(resp, "content", resp))
    assert content.strip()
