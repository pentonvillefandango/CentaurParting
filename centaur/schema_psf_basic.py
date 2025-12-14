
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

  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_basic_usable ON psf_basic_metrics(usable);
"""


def ensure_psf_basic_schema(db_path: Path) -> None:
    """
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(PSF_BASIC_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
