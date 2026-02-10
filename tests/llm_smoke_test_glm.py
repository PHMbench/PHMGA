from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _print_env_summary() -> None:
    base = os.getenv("GLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    key = os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("QUERY_GENERATOR_MODEL") or os.getenv("PHM_MODEL") or "GLM-4.7-Flash"

    print("LLM_PROVIDER=", os.getenv("LLM_PROVIDER"))
    print("MODEL=", model)
    print("BASE_HOST=", urlparse(base).netloc if base else None)
    print("KEY_SET=", bool(key), "KEY_LEN=", len(key) if key else 0)


def _http_direct_test() -> None:
    base = os.getenv("GLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    key = os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("QUERY_GENERATOR_MODEL") or "GLM-4.7-Flash"

    if not base or not key:
        raise RuntimeError("Missing GLM_API_BASE/GLM_API_KEY (or OPENAI_BASE_URL/OPENAI_API_KEY).")

    # Normalize base
    for suf in ("/chat/completions", "/v1/chat/completions"):
        if base.rstrip("/").endswith(suf):
            base = base.rstrip("/")[: -len(suf)]
    url = base.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "只回复 OK"}],
        "temperature": 0,
    }

    s = requests.Session()
    s.trust_env = False  # ignore proxies that may be misconfigured in this container
    r = s.post(url, json=payload, headers={"Authorization": f"Bearer {key}"}, timeout=30)
    print("HTTP_STATUS=", r.status_code)
    print("HTTP_BODY_HEAD=", r.text[:400].replace("\n", " "))


def _phmga_get_llm_test() -> None:
    os.environ.setdefault("LLM_PROVIDER", "glm")
    os.environ.setdefault("QUERY_GENERATOR_MODEL", "GLM-4.7-Flash")

    from src.configuration import Configuration
    from src.model import get_llm

    llm = get_llm(Configuration.from_runnable_config(None), temperature=0)
    print("PHMGA_LLM_CLASS=", llm.__class__.__name__)
    resp = llm.invoke("只回复 OK")
    print("PHMGA_LLM_REPLY=", getattr(resp, "content", resp))


def main() -> None:
    _ensure_repo_on_path()
    _load_env_file(_repo_root() / ".env")
    # Force GLM settings for this smoke test (avoid inherited env like LLM_PROVIDER=gemini).
    os.environ["LLM_PROVIDER"] = "glm"
    os.environ.setdefault("QUERY_GENERATOR_MODEL", "GLM-4.7-Flash")

    print("== ENV SUMMARY ==")
    _print_env_summary()

    print("\n== HTTP DIRECT TEST ==")
    try:
        _http_direct_test()
    except Exception as e:
        print("HTTP_TEST_ERROR=", type(e).__name__, str(e)[:250])

    print("\n== PHMGA get_llm() TEST ==")
    try:
        _phmga_get_llm_test()
    except Exception as e:
        print("PHMGA_TEST_ERROR=", type(e).__name__, str(e)[:250])


if __name__ == "__main__":
    main()
