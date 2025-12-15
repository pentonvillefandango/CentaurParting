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

  -- inputs
  n_input_stars INTEGER NOT NULL,

  -- outputs
  n_modeled INTEGER NOT NULL,

  -- gaussian summary (pixels)
  gauss_fwhm_px_median REAL,
  gauss_fwhm_px_iqr REAL,
  gauss_fwhm_px_p10 REAL,
  gauss_fwhm_px_p90 REAL,

  gauss_ecc_median REAL,
  gauss_ecc_p90 REAL,

  gauss_residual_median REAL,

  -- moffat summary (deferred/optional later)
  moffat_fwhm_px_median REAL,
  moffat_beta_median REAL,
  moffat_residual_median REAL,

  -- debug counts (stored)
  n_peaks_found INTEGER NOT NULL DEFAULT 0,
  n_fit_attempted INTEGER NOT NULL DEFAULT 0,
  n_gauss_ok INTEGER NOT NULL DEFAULT 0,

  usable INTEGER NOT NULL,
  reason TEXT,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_model_usable ON psf_model_metrics(usable);
"""

# Any column that might be missing on older DBs -> exact ADD COLUMN definition
_REQUIRED_COLS: dict[str, str] = {
    # Newer gaussian percentile outputs
    "gauss_fwhm_px_p10": "REAL",
    "gauss_fwhm_px_p90": "REAL",
    "gauss_ecc_p90": "REAL",

    # Debug counters (we store them; they’re useful for sanity-checks)
    "n_peaks_found": "INTEGER NOT NULL DEFAULT 0",
    "n_fit_attempted": "INTEGER NOT NULL DEFAULT 0",
    "n_gauss_ok": "INTEGER NOT NULL DEFAULT 0",

    # If you ever ran a DB version that had these as nullable, we still only ADD if missing.
}


def _existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r[1] for r in rows}  # r[1] = column name


def ensure_psf_model_schema(db_path: Path) -> None:
    """
    Safe to call repeatedly.
    - Creates table if missing.
    - Patches existing DBs by adding any missing columns.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1) Create base table + index (no-op if already there)
        conn.executescript(PSF_MODEL_SCHEMA_SQL)

        # 2) Patch missing columns (SQLite has no "ADD COLUMN IF NOT EXISTS")
        cols = _existing_columns(conn, "psf_model_metrics")
        for col, coldef in _REQUIRED_COLS.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE psf_model_metrics ADD COLUMN {col} {coldef};")

        conn.commit()
    finally:
        conn.close()
