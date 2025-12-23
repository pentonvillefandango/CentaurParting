# centaur/schema_dark_library_profiles.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Set


def _table_columns(con: sqlite3.Connection, table: str) -> Set[str]:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _add_column_if_missing(
    con: sqlite3.Connection, table: str, col: str, ddl: str
) -> None:
    cols = _table_columns(con, table)
    if col not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- One row per "dark library profile" the GUI can manage later.
-- Think: camera+gain+offset+temp+binning+bitdepth (+ optional readmode)
CREATE TABLE IF NOT EXISTS dark_library_profiles (
    dark_profile_id INTEGER PRIMARY KEY,

    profile_label TEXT NOT NULL,         -- e.g. "QHYMiniCam8M g80 -10C 1x1"
    camera_name TEXT NOT NULL,           -- normalized to match fits_header_core.instrume usage (lower/trim in queries)
    gain_setting INTEGER,
    offset_setting INTEGER,
    ccd_temp_c REAL,                     -- e.g. -10.0
    xbinning INTEGER,
    ybinning INTEGER,
    bitpix INTEGER,                      -- from FITS (or declared)
    readmode TEXT,                       -- optional
    notes TEXT,

    created_utc TEXT NOT NULL,
    db_updated_utc TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dark_profiles_camera_gain
  ON dark_library_profiles(camera_name, gain_setting);

CREATE INDEX IF NOT EXISTS idx_dark_profiles_temp
  ON dark_library_profiles(ccd_temp_c);
"""


def ensure_dark_library_profiles_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)

        # Forward-safe: allow adding new identity columns without refactor.
        # (kept minimal; real migrations can come later)
        _add_column_if_missing(
            con, "dark_library_profiles", "offset_setting", "INTEGER"
        )
        _add_column_if_missing(con, "dark_library_profiles", "readmode", "TEXT")

        con.commit()
    finally:
        con.close()
