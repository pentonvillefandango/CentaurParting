from __future__ import annotations

from pathlib import Path
import sqlite3


SKY_BASIC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sky_basic_metrics (
  image_id INTEGER PRIMARY KEY,
  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  exptime_s REAL,
  roi_fraction REAL NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- Full-frame level (ADU)
  ff_mean_adu REAL,
  ff_median_adu REAL,
  ff_mode_adu REAL,
  ff_sc_mean_adu REAL,
  ff_sc_median_adu REAL,
  ff_p10_adu REAL,
  ff_p25_adu REAL,
  ff_p50_adu REAL,
  ff_p75_adu REAL,
  ff_p90_adu REAL,
  ff_p99_adu REAL,
  ff_p999_adu REAL,
  ff_min_adu REAL,
  ff_max_adu REAL,

  -- Full-frame noise (ADU)
  ff_std_adu REAL,
  ff_sc_std_adu REAL,
  ff_mad_adu REAL,
  ff_madstd_adu REAL,
  ff_iqr_adu REAL,

  -- Full-frame rates (ADU/s)
  ff_median_adu_s REAL,
  ff_madstd_adu_s REAL,

  -- ROI level (ADU)
  roi_mean_adu REAL,
  roi_median_adu REAL,
  roi_mode_adu REAL,
  roi_sc_mean_adu REAL,
  roi_sc_median_adu REAL,
  roi_p10_adu REAL,
  roi_p25_adu REAL,
  roi_p50_adu REAL,
  roi_p75_adu REAL,
  roi_p90_adu REAL,
  roi_p99_adu REAL,
  roi_p999_adu REAL,
  roi_min_adu REAL,
  roi_max_adu REAL,

  -- ROI noise (ADU)
  roi_std_adu REAL,
  roi_sc_std_adu REAL,
  roi_mad_adu REAL,
  roi_madstd_adu REAL,
  roi_iqr_adu REAL,

  -- ROI rates (ADU/s)
  roi_median_adu_s REAL,
  roi_madstd_adu_s REAL,

  -- Data health
  nan_fraction REAL NOT NULL,
  inf_fraction REAL NOT NULL,
  clipped_fraction_ff REAL,
  clipped_fraction_roi REAL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_sky_basic_schema(db_path: Path) -> None:
    """
    Add Sky Basic tables if missing.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SKY_BASIC_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

