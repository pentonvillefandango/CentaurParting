from pathlib import Path
from centaur.database import Database

def ensure_exposure_advice_schema(db_path: Path) -> None:
    with Database().transaction() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS exposure_advice (
            image_id INTEGER PRIMARY KEY,
            sky_limited_min_s_k3 REAL,
            sky_limited_min_s_k5 REAL,
            gradient_limited_max_s REAL,
            recommended_min_s REAL,
            recommended_max_s REAL,
            decision_reason TEXT,
            FOREIGN KEY (image_id) REFERENCES images(image_id)
        );
        """)

