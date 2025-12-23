# centaur/schema_training_sessions.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS training_sessions (
    training_session_id INTEGER PRIMARY KEY,
    session_label TEXT,
    target_name TEXT,
    watch_root_id INTEGER,
    created_utc TEXT NOT NULL,
    started_utc TEXT,
    ended_utc TEXT,
    status TEXT NOT NULL,               -- "open" | "closed" | "aborted"
    notes TEXT,
    params_json TEXT,                  -- training plan inputs (filters, exposure candidates, etc.)
    results_json TEXT,                 -- optional snapshot (also stored in training_session_results)
    db_updated_utc TEXT NOT NULL,

    FOREIGN KEY (watch_root_id)
      REFERENCES watch_roots (watch_root_id)
);

CREATE INDEX IF NOT EXISTS idx_training_sessions_status
  ON training_sessions(status);

CREATE INDEX IF NOT EXISTS idx_training_sessions_created
  ON training_sessions(created_utc);

CREATE INDEX IF NOT EXISTS idx_training_sessions_watch_root
  ON training_sessions(watch_root_id);
"""


def ensure_training_sessions_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
