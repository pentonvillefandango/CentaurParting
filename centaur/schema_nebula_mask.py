from __future__ import annotations

from pathlib import Path
import sqlite3


NEBULA_MASK_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS nebula_mask_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  exptime_s REAL,

  -- method parameters (deterministic)
  threshold_sigma REAL NOT NULL,
  smooth_sigma_px REAL NOT NULL,
  bg_clip_sigma REAL NOT NULL,
  bg_clip_maxiters INTEGER NOT NULL,

  -- background estimate
  bg_median_adu REAL,
  bg_madstd_adu REAL,
  threshold_adu REAL,

  -- mask summary
  mask_pixel_count INTEGER,
  mask_coverage_frac REAL,

  -- connected components summary (optional; requires scipy)
  n_components INTEGER,
  largest_component_frac REAL,
  largest_component_bbox_json TEXT,

  -- data integrity
  nan_fraction REAL,
  inf_fraction REAL,

  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  usable INTEGER NOT NULL,
  reason TEXT,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_nebula_mask_schema(db_path: Path) -> None:
    """
    Add nebula mask metrics table if missing.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(NEBULA_MASK_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
