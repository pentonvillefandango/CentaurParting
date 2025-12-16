
"""
Tiny, idempotent startup migrations for Centaur Parting.

Goal: keep the DB compatible across runs without a full migration system.
Current fix: ensure module_runs.duration_us exists (INTEGER microseconds).
"""

from __future__ import annotations

import sqlite3
from typing import Set


def _table_columns(con: sqlite3.Connection, table: str) -> Set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info columns: (cid, name, type, notnull, dflt_value, pk)
    return {r[1] for r in rows}


def ensure_migrations(db_path: str) -> None:
    """
    Safe to run every startup. Makes minimal additive changes only.
    """
    con = sqlite3.connect(db_path)
    try:
        cols = _table_columns(con, "module_runs")
        if "duration_us" not in cols:
            con.execute("ALTER TABLE module_runs ADD COLUMN duration_us INTEGER")
            con.commit()
    finally:
        con.close()
