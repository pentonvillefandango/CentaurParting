from pathlib import Path
import sqlite3

def ensure_flat_metrics_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS flat_metrics (
        image_id INTEGER PRIMARY KEY,

        mean_adu REAL,
        median_adu REAL,
        std_adu REAL,
        madstd_adu REAL,

        min_adu REAL,
        max_adu REAL,

        clipped_low_frac REAL,
        clipped_high_frac REAL,

        corner_vignette_frac REAL,
        gradient_p95 REAL,

        usable INTEGER,

        db_written_utc TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
    );
    """)

    conn.commit()
    conn.close()

