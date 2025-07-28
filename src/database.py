"""Database utilities for PHM system."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

__all__ = ["SQLiteCheckpointer"]


class SQLiteCheckpointer(SqliteSaver):
    """Simple wrapper around :class:`SqliteSaver` for convenience."""

    def __init__(self, path: str) -> None:
        conn = sqlite3.connect(path, check_same_thread=False)
        super().__init__(conn)
