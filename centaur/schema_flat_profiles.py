from pathlib import Path
import sqlite3

def ensure_flat_profiles_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS flat_profiles (
        flat_profile_id INTEGER PRIMARY KEY,
        camera TEXT NOT NULL,
        filter TEXT,
        binning TEXT,
        telescope TEXT,
        focallen REAL,
        naxis1 INTEGER,
        naxis2 INTEGER,
        created_utc TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

