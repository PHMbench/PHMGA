import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _enabled() -> bool:
    return os.getenv("PHM_ENABLE_GLM_TESTS", "").strip().lower() in {"1", "true", "yes", "y"}


@pytest.mark.skipif(not _enabled(), reason="Set PHM_ENABLE_GLM_TESTS=1 to enable online GLM smoke tests.")
def test_glm_invoke_ok(monkeypatch: pytest.MonkeyPatch):
    # Ensure we are testing the real online provider, not FakeLLM from other tests.
    monkeypatch.delenv("FAKE_LLM", raising=False)
    os.environ["LLM_PROVIDER"] = "glm"
    os.environ.setdefault("QUERY_GENERATOR_MODEL", "GLM-4.7-Flash")

    from src.configuration import Configuration
    from src.model import get_llm

    try:
        llm = get_llm(Configuration.from_runnable_config(None), temperature=0)
    except ValueError as e:
        # Accept `.env`-driven setups; if still missing key/base, skip instead of failing CI.
        msg = str(e).lower()
        if "missing" in msg and ("api key" in msg or "base_url" in msg or "base url" in msg):
            pytest.skip(str(e))
        raise
    assert llm.__class__.__name__ == "ChatOpenAI"

    resp = llm.invoke("只回复 OK")
    content = getattr(resp, "content", str(resp))
    assert "OK" in str(content)
