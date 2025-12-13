from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from centaur.init_db import DEFAULT_DB_PATH


@dataclass(frozen=True)
class DbConfig:
    db_path: Path = DEFAULT_DB_PATH
    busy_timeout_ms: int = 5000  # wait up to 5s if DB is busy
    enable_wal: bool = True


class Database:
    """
    SQLite access layer.
    - One connection per Database instance (we'll use a single writer instance in v1)
    - Provides safe execute helpers and transaction context manager
    - Contains NO business logic (no FITS parsing, no watcher logic)
    """

    def __init__(self, config: DbConfig = DbConfig()) -> None:
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        if self._conn is not None:
            return

        self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._config.db_path)

        # Sensible defaults
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {int(self._config.busy_timeout_ms)};")

        # WAL improves concurrency (multiple readers + one writer)
        if self._config.enable_wal:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")

        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    def executescript(self, script: str) -> None:
        self.conn.executescript(script)
        self.conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, seq_of_params: Iterable[tuple[Any, ...]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, seq_of_params)

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def transaction(self):
        """
        Context manager for a transaction:
        - commits if no exception
        - rolls back on error
        """
        return _Transaction(self)


class _Transaction:
    def __init__(self, db: Database) -> None:
        self._db = db

    def __enter__(self) -> Database:
        # BEGIN IMMEDIATE helps avoid deadlocks and makes "who owns the write lock" explicit
        self._db.execute("BEGIN IMMEDIATE;")
        return self._db

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._db.commit()
        else:
            self._db.rollback()

