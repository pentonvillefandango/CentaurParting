# centaur/schema_training_derived_metrics.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- One derived row per image; this is the "feature layer" used by training analysis.
-- Filled by a worker (later): training_derived_worker.
CREATE TABLE IF NOT EXISTS training_derived_metrics (
    image_id INTEGER PRIMARY KEY,

    expected_fields INTEGER NOT NULL,
    read_fields INTEGER NOT NULL,
    written_fields INTEGER NOT NULL,
    parse_warnings TEXT,
    db_written_utc TEXT NOT NULL,

    -- Identity / grouping keys
    filter TEXT,
    exptime_s REAL,
    camera_name TEXT,
    gain_setting INTEGER,

    -- Key training signals (pulled from existing metric tables)
    transparency_proxy REAL,
    headroom_p99 REAL,
    star_peak_rate_p99_adu_s REAL,
    nebula_minus_bg_adu_s REAL,
    sky_ff_median_adu_s REAL,
    sky_limited_ratio REAL,

    -- Ratios used in your SQL experiments
    nebula_over_sky REAL,               -- nebula_minus_bg_adu_s / sky_ff_median_adu_s
    eff_proxy REAL,                     -- example: nebula_minus_bg_adu_s / sqrt(sky_ff_median_adu_s)

    -- Constraints / safety signals
    effective_ceiling_adu REAL,
    star_peak_p99_adu REAL,
    p99_over_linear_ceiling REAL,
    linear_headroom_p99 REAL,

    -- Optional conditions linkage (if available)
    airmass REAL,
    moon_alt_deg REAL,
    moon_illum_frac REAL,
    moon_sep_deg REAL,

    usable INTEGER NOT NULL,
    reason TEXT,

    FOREIGN KEY (image_id)
      REFERENCES images (image_id)
      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_training_derived_filter_exp
  ON training_derived_metrics(filter, exptime_s);

CREATE INDEX IF NOT EXISTS idx_training_derived_usable
  ON training_derived_metrics(usable);
"""


def ensure_training_derived_metrics_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
