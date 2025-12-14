
from __future__ import annotations

import sqlite3
from pathlib import Path


def ensure_flat_links_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS flat_frame_links (
        image_id INTEGER PRIMARY KEY,
        flat_profile_id INTEGER NOT NULL,
        flat_capture_set_id INTEGER NOT NULL,
        created_utc TEXT NOT NULL,
        FOREIGN KEY(image_id) REFERENCES images(image_id),
        FOREIGN KEY(flat_profile_id) REFERENCES flat_profiles(flat_profile_id),
        FOREIGN KEY(flat_capture_set_id) REFERENCES flat_capture_sets(flat_capture_set_id)
    );
    """)

    conn.commit()
    conn.close()
