from __future__ import annotations

from pathlib import Path
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

__all__ = ["SQLiteCheckpointer"]

class SQLiteCheckpointer:
    """Wrapper around :class:`SqliteSaver` for persistent graph state."""

    def __init__(self, path: str | Path) -> None:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        self.saver = SqliteSaver(conn)

    def __getattr__(self, name):
        return getattr(self.saver, name)
