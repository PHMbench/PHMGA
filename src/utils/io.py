"""Small IO helpers for artifact writing and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create a directory tree if needed and return it as a ``Path``."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(payload: Any, path: Union[str, Path]) -> Path:
    """Write a JSON artifact with UTF-8 encoding and parent directory creation."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return target


def write_text(content: str, path: Union[str, Path]) -> Path:
    """Write a text artifact with UTF-8 encoding and parent directory creation."""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        handle.write(content)
    return target


def hash_payload(payload: Any) -> str:
    """Hash a JSON-serializable payload for manifest stability."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
