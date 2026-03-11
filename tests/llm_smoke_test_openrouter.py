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
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _print_env_summary() -> None:
    base = os.getenv("OPENROUTER_BASE_URL")
    key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("QUERY_GENERATOR_MODEL")
    print("LLM_PROVIDER=", os.getenv("LLM_PROVIDER"))
    print("MODEL=", model)
    print("BASE_HOST=", urlparse(base).netloc if base else None)
    print("KEY_SET=", bool(key), "KEY_LEN=", len(key) if key else 0)


def _phmga_get_llm_test() -> None:
    from src.model import get_llm

    llm = get_llm(temperature=0)
    print("PHMGA_LLM_CLASS=", llm.__class__.__name__)
    resp = llm.invoke("Reply exactly OK")
    content = getattr(resp, "content", resp)
    print("PHMGA_LLM_REPLY=", content)


def main() -> None:
    _ensure_repo_on_path()
    _load_env_file(_repo_root() / ".env")
    os.environ.setdefault("LLM_PROVIDER", "openrouter")
    os.environ.setdefault("QUERY_GENERATOR_MODEL", "openai/gpt-4o-mini")

    print("== ENV SUMMARY ==")
    _print_env_summary()

    print("\n== PHMGA get_llm() OPENROUTER TEST ==")
    try:
        _phmga_get_llm_test()
    except Exception as exc:
        print("PHMGA_TEST_ERROR=", type(exc).__name__, str(exc)[:400])


if __name__ == "__main__":
    main()
