# centaur/schema_dark_library_exposures.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Declares which exposure lengths are supported for a given dark profile.
-- This is what the advice engine will constrain recommendations to.
CREATE TABLE IF NOT EXISTS dark_library_exposures (
    dark_profile_id INTEGER NOT NULL,
    exptime_s REAL NOT NULL,

    -- Inventory / provenance
    n_dark_frames INTEGER,              -- count in the library (optional but useful)
    has_master_dark INTEGER NOT NULL,   -- 0/1
    master_dark_path TEXT,              -- optional
    notes TEXT,

    created_utc TEXT NOT NULL,
    db_updated_utc TEXT NOT NULL,

    PRIMARY KEY (dark_profile_id, exptime_s),

    FOREIGN KEY (dark_profile_id)
      REFERENCES dark_library_profiles (dark_profile_id)
      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dark_exposures_exptime
  ON dark_library_exposures(exptime_s);
"""


def ensure_dark_library_exposures_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
