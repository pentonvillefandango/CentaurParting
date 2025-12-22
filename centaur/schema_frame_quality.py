from __future__ import annotations

from pathlib import Path
import sqlite3


FRAME_QUALITY_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frame_quality_metrics (
  image_id INTEGER PRIMARY KEY,

  -- audit
  expected_fields INTEGER NOT NULL,
  read_fields INTEGER NOT NULL,
  written_fields INTEGER NOT NULL,
  parse_warnings TEXT,
  db_written_utc TEXT NOT NULL,

  -- core outputs
  quality_score INTEGER NOT NULL,          -- 0..100
  decision TEXT NOT NULL,                  -- KEEP|WARN|REJECT
  reason_mask INTEGER NOT NULL,
  primary_reason TEXT NOT NULL,

  -- subscores (0..100)
  psf_score INTEGER NOT NULL,
  bg_score INTEGER NOT NULL,
  clip_score INTEGER NOT NULL,

  -- NEW subscores (0..100)
  signal_score INTEGER NOT NULL,
  headroom_score INTEGER NOT NULL,
  confidence_score INTEGER NOT NULL,

  -- driver fields copied for convenience/plotting
  fwhm_px_median REAL,
  fwhm_px_p10 REAL,
  fwhm_px_p90 REAL,
  ecc_median REAL,
  ecc_p90 REAL,
  n_measured INTEGER,

  saturated_pixel_fraction REAL,

  bkg2d_rms_of_map_adu REAL,
  plane_slope_mag_adu_per_tile REAL,
  grad_p95_adu_per_tile REAL,
  corner_delta_adu REAL,

  -- NEW driver fields (signal/headroom)
  obj_minus_bg_adu_s REAL,
  nebula_minus_bg_adu_s REAL,
  headroom_p99 REAL,

  usable INTEGER NOT NULL,
  reason TEXT NOT NULL,

  FOREIGN KEY (image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_frame_quality_decision ON frame_quality_metrics(decision);
CREATE INDEX IF NOT EXISTS idx_frame_quality_score ON frame_quality_metrics(quality_score);
"""


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, col_name: str, col_def: str
) -> None:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    if col_name in cols:
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")


def ensure_frame_quality_schema(db_path: Path) -> None:
    """
    Safe to call repeatedly.
    - Creates frame_quality_metrics if missing
    - Adds upgrade columns if upgrading an existing DB
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(FRAME_QUALITY_SCHEMA_SQL)

        # Upgrade path for existing DBs (ADD COLUMN is idempotent via check)
        upgrades = [
            ("signal_score", "INTEGER"),
            ("headroom_score", "INTEGER"),
            ("confidence_score", "INTEGER"),
            ("fwhm_px_p10", "REAL"),
            ("fwhm_px_p90", "REAL"),
            ("ecc_p90", "REAL"),
            ("obj_minus_bg_adu_s", "REAL"),
            ("nebula_minus_bg_adu_s", "REAL"),
            ("headroom_p99", "REAL"),
        ]
        for col, coldef in upgrades:
            _add_column_if_missing(conn, "frame_quality_metrics", col, coldef)

        conn.commit()
    finally:
        conn.close()
