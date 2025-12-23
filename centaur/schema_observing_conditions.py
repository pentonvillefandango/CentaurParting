# centaur/schema_observing_conditions.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Per-image "conditions" row, derived from header + optional astropy calcs.
-- Filled by a worker (later): observing_conditions_worker.
CREATE TABLE IF NOT EXISTS observing_conditions (
    image_id INTEGER PRIMARY KEY,

    expected_fields INTEGER NOT NULL,
    read_fields INTEGER NOT NULL,
    written_fields INTEGER NOT NULL,
    parse_warnings TEXT,
    db_written_utc TEXT NOT NULL,

    -- Location/time inputs (from header where possible)
    date_obs TEXT,
    latitude REAL,
    longitude REAL,
    elevation_m REAL,

    -- Derived
    airmass REAL,
    alt_deg REAL,
    az_deg REAL,
    sun_alt_deg REAL,
    moon_alt_deg REAL,
    moon_illum_frac REAL,
    moon_sep_deg REAL,

    -- Basic "night context"
    filter TEXT,
    exptime_s REAL,
    camera_name TEXT,

    usable INTEGER NOT NULL,
    reason TEXT,

    FOREIGN KEY (image_id)
      REFERENCES images (image_id)
      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_observing_conditions_usable
  ON observing_conditions(usable);

CREATE INDEX IF NOT EXISTS idx_observing_conditions_date
  ON observing_conditions(date_obs);
"""


def ensure_observing_conditions_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
