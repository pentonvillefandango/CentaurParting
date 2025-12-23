# centaur/schema_training_session_frames.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS training_session_frames (
    training_session_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,

    -- Optional: classify frames inside the session
    role TEXT,                          -- e.g. "train" | "validate" | "monitor"
    filter TEXT,                        -- captured filter at ingest time (redundant convenience)
    exptime_s REAL,                     -- redundant convenience
    gain_setting INTEGER,               -- redundant convenience
    tagged_utc TEXT NOT NULL,
    tagged_by TEXT,                     -- e.g. "cli" | "auto"

    PRIMARY KEY (training_session_id, image_id),

    FOREIGN KEY (training_session_id)
      REFERENCES training_sessions (training_session_id)
      ON DELETE CASCADE,

    FOREIGN KEY (image_id)
      REFERENCES images (image_id)
      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tsf_image
  ON training_session_frames(image_id);

CREATE INDEX IF NOT EXISTS idx_tsf_session
  ON training_session_frames(training_session_id);

CREATE INDEX IF NOT EXISTS idx_tsf_role
  ON training_session_frames(role);
"""


def ensure_training_session_frames_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
