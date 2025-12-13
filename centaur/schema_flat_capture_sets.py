from pathlib import Path
import sqlite3

def ensure_flat_capture_sets_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS flat_capture_sets (
        flat_capture_set_id INTEGER PRIMARY KEY,
        night TEXT NOT NULL,
        target TEXT,
        camera TEXT NOT NULL,
        filter TEXT,
        binning TEXT,
        exptime REAL,
        flat_profile_id INTEGER NOT NULL,
        created_utc TEXT NOT NULL,
        FOREIGN KEY(flat_profile_id) REFERENCES flat_profiles(flat_profile_id)
    );
    """)

    conn.commit()
    conn.close()

