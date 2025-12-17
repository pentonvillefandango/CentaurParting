from __future__ import annotations

from pathlib import Path
import sqlite3

MASKED_SIGNAL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS masked_signal_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  exptime_s REAL,

  -- pixel counts
  nebula_pixel_count INTEGER,
  bg_pixel_count INTEGER,
  nebula_frac REAL,

  -- signal stats
  nebula_median_adu REAL,
  nebula_madstd_adu REAL,
  bg_median_adu REAL,
  bg_madstd_adu REAL,

  -- contrast
  nebula_minus_bg_adu REAL,
  nebula_minus_bg_adu_s REAL,

  -- SNR proxy (dimensionless)
  snr_proxy REAL,
  snr_proxy_s REAL,

  -- integrity
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  usable INTEGER NOT NULL,
  reason TEXT,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_masked_signal_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(MASKED_SIGNAL_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
