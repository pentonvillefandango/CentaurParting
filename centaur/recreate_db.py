# centaur/recreate_db.py
from __future__ import annotations

import argparse
from pathlib import Path

from centaur.init_db import DEFAULT_DB_PATH, init_db


def recreate_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    if db_path.exists():
        db_path.unlink()
    init_db(db_path=db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DEV ONLY: delete and recreate the Centaur Parting database.")
    parser.add_argument(
        "--yes-really",
        action="store_true",
        help="Required confirmation flag. Without this, nothing happens.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to the sqlite database file (default: data/centaurparting.db).",
    )
    args = parser.parse_args()

    if not args.yes_really:
        print("Refusing to recreate DB. Re-run with: --yes-really")
        return

    recreate_db(db_path=Path(args.db))


if __name__ == "__main__":
    main()

