"""Minimal .env loading for provider credentials."""

from __future__ import annotations

import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def _find_dotenv(start_dir: Path) -> Path | None:
    resolved = start_dir.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    for candidate_dir in (resolved, *resolved.parents):
        candidate = candidate_dir / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_runtime_dotenv(env_file: str | None = None, *, start_dir: Path | None = None) -> Path | None:
    """Load credentials from .env without overriding process environment values."""

    if env_file == "":
        return None

    if env_file:
        path = Path(env_file).expanduser()
        if not path.is_absolute() and start_dir is not None:
            path = start_dir / path
    else:
        path = _find_dotenv(start_dir or Path.cwd())

    if path is None or not path.is_file():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        os.environ.setdefault(key, value)
    return path


__all__ = ["load_runtime_dotenv"]
