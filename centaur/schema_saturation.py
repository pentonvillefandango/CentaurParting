
from __future__ import annotations

from pathlib import Path

import sqlite3


def ensure_saturation_schema(db_path: Path) -> None:
    """
    Create saturation_metrics table (safe to run repeatedly).
    One row per image_id.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS saturation_metrics (
                image_id INTEGER PRIMARY KEY,

                expected_fields INTEGER,
                read_fields INTEGER,
                written_fields INTEGER,

                exptime_s REAL,

                -- Logging / audit
                parse_warnings TEXT,
                db_written_utc TEXT NOT NULL,

                -- Core saturation / headroom
                saturation_adu REAL,                  -- optional threshold used (NULL allowed)
                max_pixel_adu REAL NOT NULL,
                saturated_pixel_count INTEGER NOT NULL,
                saturated_pixel_fraction REAL NOT NULL,

                -- Optional: brightest detected star peak (if computed)
                brightest_star_peak_adu REAL,

                -- Numeric pathologies
                nan_fraction REAL,
                inf_fraction REAL,

                usable INTEGER,
                reason TEXT,

                FOREIGN KEY (image_id)
                    REFERENCES images (image_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_saturation_metrics_image
                ON saturation_metrics (image_id);
            """
        )
        conn.commit()
    finally:
        conn.close()
