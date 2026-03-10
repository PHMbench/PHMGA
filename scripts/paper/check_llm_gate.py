#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _load_local_dotenv() -> None:
    try:
        load_dotenv(dotenv_path=str(Path.cwd() / ".env"), override=False)
    except Exception:
        pass


def _gate_a1(provider: str, model: str) -> int:
    if provider.strip().lower() != "glm":
        print(f"GATE_A1_FAIL unsupported provider for zai gate: {provider!r}")
        return 2
    try:
        from zai import ZhipuAiClient
    except Exception as exc:
        print(f"GATE_A1_FAIL zai import error: {type(exc).__name__}")
        return 3

    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        print("GATE_A1_FAIL missing GLM_API_KEY")
        return 4

    client = ZhipuAiClient(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=str(model).strip().lower(),
            messages=[{"role": "user", "content": "只回复 OK"}],
            thinking={"type": "enabled"},
            stream=True,
            max_tokens=128,
            temperature=0.1,
        )
    except Exception as exc:
        print(f"GATE_A1_FAIL request error: {type(exc).__name__}: {exc}")
        return 5

    chunks: list[str] = []
    seen = False
    for chunk in response:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        if reasoning:
            seen = True
            chunks.append(str(reasoning))
        if content:
            seen = True
            chunks.append(str(content))

    text = "".join(chunks)
    print(f"GATE_A1_OUTPUT={text[:240]}")
    if not seen:
        print("GATE_A1_FAIL empty streaming response")
        return 6
    if "OK" not in text.upper():
        print("GATE_A1_FAIL response does not contain OK")
        return 7

    print("GATE_A1_PASS")
    return 0


def _gate_a2(config_path: str) -> int:
    from src.cases.case1 import _bind_llm_from_case
    from src.model import get_llm

    cfg_path = Path(config_path).expanduser().resolve()
    if not cfg_path.exists():
        print(f"GATE_A2_FAIL config not found: {cfg_path}")
        return 2

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        _bind_llm_from_case(cfg)
    except Exception as exc:
        print(f"GATE_A2_FAIL bind llm: {type(exc).__name__}: {exc}")
        return 3

    try:
        resp = get_llm().invoke("只回复 OK")
    except Exception as exc:
        print(f"GATE_A2_FAIL invoke: {type(exc).__name__}: {exc}")
        return 4
    txt = getattr(resp, "content", str(resp))
    print(f"GATE_A2_OUTPUT={txt}")
    if "OK" not in str(txt).upper():
        print("GATE_A2_FAIL response does not contain OK")
        return 5

    print("GATE_A2_PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified LLM connectivity gates with dotenv loading.")
    parser.add_argument("--gate", required=True, choices=["a1", "a2"])
    parser.add_argument("--provider", default="glm")
    parser.add_argument("--model", default="glm-4.7-flash")
    parser.add_argument("--config", default="")
    args = parser.parse_args()

    _load_local_dotenv()

    if args.gate == "a1":
        return _gate_a1(provider=args.provider, model=args.model)
    if not args.config:
        print("GATE_A2_FAIL --config is required for gate a2")
        return 2
    return _gate_a2(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

