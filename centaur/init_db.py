# centaur/init_db.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from centaur.schema import SCHEMA_SQL


DEFAULT_DB_PATH = Path("data") / "centaurparting.db"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        # Apply schema in one go
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()

    print(f"DB initialized: {db_path.resolve()}")


if __name__ == "__main__":
    init_db()

