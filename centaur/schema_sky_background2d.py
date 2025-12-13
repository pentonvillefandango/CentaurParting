from __future__ import annotations

from pathlib import Path
import sqlite3


SKY_BKG2D_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sky_background2d_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  exptime_s REAL,
  tile_size_px INTEGER NOT NULL,
  grid_nx INTEGER NOT NULL,
  grid_ny INTEGER NOT NULL,
  clipped_fraction_mean REAL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- Background map summary (ADU)
  bkg2d_median_adu REAL,
  bkg2d_min_adu REAL,
  bkg2d_max_adu REAL,
  bkg2d_range_adu REAL,
  bkg2d_p95_minus_p5_adu REAL,
  bkg2d_rms_of_map_adu REAL,

  -- Plane fit slopes (ADU per tile index)
  plane_slope_x_adu_per_tile REAL,
  plane_slope_y_adu_per_tile REAL,
  plane_slope_mag_adu_per_tile REAL,

  -- Gradient magnitude over the tile grid (ADU per tile)
  grad_mean_adu_per_tile REAL,
  grad_p95_adu_per_tile REAL,

  -- Simple corner metric
  corner_delta_adu REAL,

  -- Per-second variants where exptime is known
  bkg2d_median_adu_s REAL,
  plane_slope_mag_adu_per_tile_s REAL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def ensure_sky_background2d_schema(db_path: Path) -> None:
    """
    Add Background2D tables if missing.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SKY_BKG2D_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

