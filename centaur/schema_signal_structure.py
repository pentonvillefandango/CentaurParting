
from __future__ import annotations

from pathlib import Path
import sqlite3


SIGNAL_STRUCTURE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signal_structure_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- provenance / inputs
  exptime_s REAL,

  -- structure (ADU) computed from sky_basic percentiles
  ff_p90_minus_p10_adu REAL,
  ff_p99_minus_p50_adu REAL,
  ff_p999_minus_p50_adu REAL,

  -- per-second structure (ADU/s)
  ff_p90_minus_p10_adu_s REAL,
  ff_p99_minus_p50_adu_s REAL,
  ff_p999_minus_p50_adu_s REAL,

  -- noise and gradients (per-second)
  ff_madstd_adu_s REAL,
  plane_slope_mag_adu_per_tile_s REAL,

  -- other quality indicators
  saturated_pixel_fraction REAL,
  psf_fwhm_px_median REAL,
  psf_ecc_median REAL,
  psf_usable INTEGER,

  -- derived decision helpers
  eff_score REAL,
  time_weight REAL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_signal_structure_schema(db_path: Path) -> None:
    """
    Add signal_structure_metrics table if missing.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SIGNAL_STRUCTURE_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
