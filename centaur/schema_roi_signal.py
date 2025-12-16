
from __future__ import annotations

from pathlib import Path

import sqlite3


def ensure_roi_signal_schema(db_path: Path) -> None:
    """
    Create roi_signal_metrics table (safe to run repeatedly).
    One row per image_id.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS roi_signal_metrics (
                image_id INTEGER PRIMARY KEY,

                expected_fields INTEGER,
                read_fields INTEGER,
                written_fields INTEGER,

                exptime_s REAL,

                -- ROI definition (deterministic, one row per image)
                roi_kind TEXT NOT NULL,          -- e.g. 'target_v1'
                roi_fraction REAL,
                roi_bbox_json TEXT,              -- e.g. {"x0":..,"y0":..,"x1":..,"y1":..}
                bg_bbox_json TEXT,               -- optional local BG region definition

                -- Measurements (ADU)
                obj_median_adu REAL,
                obj_madstd_adu REAL,
                bg_median_adu REAL,
                bg_madstd_adu REAL,

                -- Background-subtracted signal proxy
                obj_minus_bg_adu REAL,
                obj_minus_bg_adu_s REAL,

                -- Optional: clipping fractions within obj/bg regions
                clipped_fraction_obj REAL,
                clipped_fraction_bg REAL,

                -- Logging / audit
                parse_warnings TEXT,
                db_written_utc TEXT NOT NULL,

                usable INTEGER,
                reason TEXT,

                FOREIGN KEY (image_id)
                    REFERENCES images (image_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_roi_signal_metrics_image
                ON roi_signal_metrics (image_id);

            CREATE INDEX IF NOT EXISTS idx_roi_signal_metrics_kind
                ON roi_signal_metrics (roi_kind);
            """
        )
        conn.commit()
    finally:
        conn.close()
