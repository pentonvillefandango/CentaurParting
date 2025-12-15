
from __future__ import annotations

from pathlib import Path
import sqlite3


PSF_MODEL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS psf_model_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  db_written_utc TEXT NOT NULL,

  -- inputs / counts
  n_input_stars INTEGER NOT NULL,
  n_modeled INTEGER NOT NULL,

  -- NEW (PSF-2 internal denominators)
  n_peaks_found INTEGER NOT NULL,
  n_fit_attempted INTEGER NOT NULL,
  n_gauss_ok INTEGER NOT NULL,

  -- gaussian summary
  gauss_fwhm_px_median REAL,
  gauss_fwhm_px_iqr REAL,
  gauss_ecc_median REAL,
  gauss_residual_median REAL,

  -- moffat placeholders (future)
  moffat_fwhm_px_median REAL,
  moffat_beta_median REAL,
  moffat_residual_median REAL,

  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);
"""


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_name: str, col_def: str) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if col_name in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")


def ensure_psf_model_metrics_schema(db_path: Path) -> None:
    """
    Safe to call repeatedly.
    - Creates table if missing
    - Adds new columns if missing (ALTER TABLE)
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(PSF_MODEL_SCHEMA_SQL)

        # If table already existed from older versions, ensure new columns exist.
        _add_column_if_missing(conn, "psf_model_metrics", "n_peaks_found", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "psf_model_metrics", "n_fit_attempted", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "psf_model_metrics", "n_gauss_ok", "INTEGER NOT NULL DEFAULT 0")

        conn.commit()
    finally:
        conn.close()
