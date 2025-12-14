
from __future__ import annotations

from pathlib import Path
import sqlite3


PSF_DETECT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS psf_detect_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- config / runtime
  roi_fraction REAL NOT NULL,
  threshold_sigma REAL NOT NULL,
  min_separation_px INTEGER NOT NULL,
  max_stars INTEGER NOT NULL,

  -- background estimate used for detection
  bg_median_adu REAL,
  bg_madstd_adu REAL,
  threshold_adu REAL,

  -- legacy counts (kept for compatibility)
  n_candidates_total INTEGER NOT NULL,
  n_candidates_used INTEGER NOT NULL,
  n_saturated_candidates INTEGER NOT NULL,
  n_edge_rejected INTEGER NOT NULL,

  -- health
  nan_fraction REAL NOT NULL,
  inf_fraction REAL NOT NULL,

  -- usability
  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_psf_detect_usable ON psf_detect_metrics(usable);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """
    SQLite-safe 'add column if missing' helper.
    """
    existing = set()
    for row in conn.execute(f"PRAGMA table_info({table});").fetchall():
        existing.add(row[1])

    if col in existing:
        return

    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl};")


def ensure_psf_detect_schema(db_path: Path) -> None:
    """
    Add PSF Detect tables if missing, and apply additive column upgrades.
    Safe to call repeatedly.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(PSF_DETECT_SCHEMA_SQL)

        # Additive v2 columns (no DB reset required)
        _ensure_column(conn, "psf_detect_metrics", "n_peaks_total", "INTEGER")
        _ensure_column(conn, "psf_detect_metrics", "n_peaks_good", "INTEGER")
        _ensure_column(conn, "psf_detect_metrics", "peak_window", "INTEGER")
        _ensure_column(conn, "psf_detect_metrics", "good_extra_sigma", "REAL")
        _ensure_column(conn, "psf_detect_metrics", "good_threshold_adu", "REAL")

        conn.commit()
    finally:
        conn.close()
