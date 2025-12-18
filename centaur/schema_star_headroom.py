from __future__ import annotations

from pathlib import Path
import sqlite3


STAR_HEADROOM_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS star_headroom_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  exptime_s REAL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- stars used
  n_stars_used INTEGER,

  -- saturation reference actually used for headroom math
  saturation_adu_used REAL,
  saturation_source TEXT,

  -- peak ADU percentiles across stars (from star peaks)
  star_peak_p50_adu REAL,
  star_peak_p90_adu REAL,
  star_peak_p99_adu REAL,

  -- headroom relative to saturation_adu_used, in [0..1]
  headroom_p50 REAL,
  headroom_p90 REAL,
  headroom_p99 REAL,

  -- fraction of stars considered saturated (peak >= saturation_adu_used - eps)
  saturated_star_fraction REAL,

  usable INTEGER NOT NULL,
  reason TEXT,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_star_headroom_schema(db_path: Path) -> None:
    """
    Add star headroom metrics table if missing.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(STAR_HEADROOM_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
