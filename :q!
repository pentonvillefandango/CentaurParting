from pathlib import Path
from centaur.database import Database

def ensure_camera_constants_schema(db_path: Path) -> None:
    with Database().transaction() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS camera_constants (
            camera_name TEXT NOT NULL,
            gain_setting INTEGER,
            gain_e_per_adu REAL NOT NULL,
            read_noise_e REAL NOT NULL,
            is_osc INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            PRIMARY KEY (camera_name, gain_setting)
        );
        """)

