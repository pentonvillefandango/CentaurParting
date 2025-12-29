from pathlib import Path
import sqlite3


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(str(r[1]) == col for r in rows)  # r[1] is column name


def ensure_flat_metrics_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table if missing (new installs get the full schema)
    cur.execute(
        """
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

        -- Bounded, normalized [0,1] metric used downstream
        corner_vignette_frac REAL,

        -- NEW: raw ratio corner_med/median (unclamped)
        corner_vignette_ratio_raw REAL,

        gradient_p95 REAL,

        usable INTEGER,

        db_written_utc TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
    );
    """
    )

    # Migrate existing DBs: add the column if the table predates it.
    if not _column_exists(conn, "flat_metrics", "corner_vignette_ratio_raw"):
        cur.execute(
            "ALTER TABLE flat_metrics ADD COLUMN corner_vignette_ratio_raw REAL;"
        )

    conn.commit()
    conn.close()
