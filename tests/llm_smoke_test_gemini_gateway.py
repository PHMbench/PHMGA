from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import urlparse


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


def _prepare_gateway_env() -> None:
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


def _print_env_summary() -> None:
    base = os.getenv("OPENAI_BASE_URL")
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("QUERY_GENERATOR_MODEL")
    print("LLM_PROVIDER=", os.getenv("LLM_PROVIDER"))
    print("MODEL=", model)
    print("BASE_HOST=", urlparse(base).netloc if base else None)
    print("KEY_SET=", bool(key), "KEY_LEN=", len(key) if key else 0)


def _phmga_get_llm_test() -> None:
    from src.configuration import Configuration
    from src.model import get_llm

    llm = get_llm(Configuration.from_runnable_config(None), temperature=0)
    print("PHMGA_LLM_CLASS=", llm.__class__.__name__)
    resp = llm.invoke("Reply exactly OK")
    content = getattr(resp, "content", resp)
    print("PHMGA_LLM_REPLY=", content)


def main() -> None:
    _ensure_repo_on_path()
    _load_env_file(_repo_root() / ".env")
    _prepare_gateway_env()

    print("== ENV SUMMARY ==")
    _print_env_summary()

    print("\n== PHMGA get_llm() GATEWAY TEST ==")
    try:
        _phmga_get_llm_test()
    except Exception as e:
        print("PHMGA_TEST_ERROR=", type(e).__name__, str(e)[:400])


if __name__ == "__main__":
    main()
