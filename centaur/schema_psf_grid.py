from __future__ import annotations

from pathlib import Path
import sqlite3


PSF_GRID_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS psf_grid_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- grid definition
  grid_rows INTEGER NOT NULL,
  grid_cols INTEGER NOT NULL,
  grid_min_stars_per_cell INTEGER NOT NULL,

  -- overall counts
  n_input_stars INTEGER NOT NULL,
  n_cells_with_data INTEGER NOT NULL,

  -- rollups
  center_fwhm_px REAL,
  corner_fwhm_px_median REAL,
  center_to_corner_fwhm_ratio REAL,
  left_right_fwhm_ratio REAL,
  top_bottom_fwhm_ratio REAL,

  -- per-cell summaries (3x3)
  cell_r0c0_n INTEGER NOT NULL,
  cell_r0c0_fwhm_px_median REAL,
  cell_r0c0_ecc_median REAL,

  cell_r0c1_n INTEGER NOT NULL,
  cell_r0c1_fwhm_px_median REAL,
  cell_r0c1_ecc_median REAL,

  cell_r0c2_n INTEGER NOT NULL,
  cell_r0c2_fwhm_px_median REAL,
  cell_r0c2_ecc_median REAL,

  cell_r1c0_n INTEGER NOT NULL,
  cell_r1c0_fwhm_px_median REAL,
  cell_r1c0_ecc_median REAL,

  cell_r1c1_n INTEGER NOT NULL,
  cell_r1c1_fwhm_px_median REAL,
  cell_r1c1_ecc_median REAL,

  cell_r1c2_n INTEGER NOT NULL,
  cell_r1c2_fwhm_px_median REAL,
  cell_r1c2_ecc_median REAL,

  cell_r2c0_n INTEGER NOT NULL,
  cell_r2c0_fwhm_px_median REAL,
  cell_r2c0_ecc_median REAL,

  cell_r2c1_n INTEGER NOT NULL,
  cell_r2c1_fwhm_px_median REAL,
  cell_r2c1_ecc_median REAL,

  cell_r2c2_n INTEGER NOT NULL,
  cell_r2c2_fwhm_px_median REAL,
  cell_r2c2_ecc_median REAL,

  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_grid_usable ON psf_grid_metrics(usable);
"""


def ensure_psf_grid_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(PSF_GRID_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
