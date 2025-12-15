from __future__ import annotations

from pathlib import Path
import sqlite3


PSF_BASIC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS psf_basic_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- selection / counts
  roi_fraction REAL NOT NULL,
  threshold_sigma REAL NOT NULL,
  good_extra_sigma REAL NOT NULL,
  min_separation_px INTEGER NOT NULL,
  edge_margin_px INTEGER NOT NULL,
  cutout_radius_px INTEGER NOT NULL,
  max_stars_measured INTEGER NOT NULL,

  n_peaks_total INTEGER NOT NULL,
  n_peaks_good INTEGER NOT NULL,
  n_measured INTEGER NOT NULL,
  n_rejected_cutout INTEGER NOT NULL,
  n_rejected_measure INTEGER NOT NULL,

  -- HFR (pixels)
  hfr_px_median REAL,
  hfr_px_p10 REAL,
  hfr_px_p90 REAL,

  -- FWHM proxy (pixels)
  fwhm_px_median REAL,
  fwhm_px_p10 REAL,
  fwhm_px_p90 REAL,

  -- Eccentricity
  ecc_median REAL,
  ecc_p90 REAL,

  -- Optional angle summary (radians)
  theta_rad_median REAL,
  theta_rad_p90abs REAL,

  -- NEW: PSF-1 accepted star list (JSON)
  -- Example: [{"x":123,"y":456},{"x":...,"y":...}]
  star_xy_json TEXT,

  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_basic_usable ON psf_basic_metrics(usable);
"""


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_name: str, col_def: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if col_name in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")


def ensure_psf_basic_schema(db_path: Path) -> None:
    """
    Safe to call repeatedly.
    - Creates psf_basic_metrics if missing
    - Adds star_xy_json if upgrading an existing DB
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(PSF_BASIC_SCHEMA_SQL)

        # Upgrade path for existing DBs
        _add_column_if_missing(conn, "psf_basic_metrics", "star_xy_json", "TEXT")

        conn.commit()
    finally:
        conn.close()
