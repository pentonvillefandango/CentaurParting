from __future__ import annotations

from centaur.database import Database


def seed_camera_constants() -> None:
    """
    Seed known camera constants into the DB.
    Safe to run repeatedly (INSERT OR REPLACE).
    """
    with Database().transaction() as db:
        db.execute(
            """
            INSERT OR REPLACE INTO camera_constants
            (camera_name, gain_setting, gain_e_per_adu, read_noise_e, is_osc, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("qhyminicam8m", 80, 0.82, 1.07, 0, "SharpCap analysis (12-bit)"),
        )

        db.execute(
            """
            INSERT OR REPLACE INTO camera_constants
            (camera_name, gain_setting, gain_e_per_adu, read_noise_e, is_osc, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("zwo asi585mc pro", 252, 1.0, 1.0, 1, "Unity gain (SharpCap + ZWO curves)"),
        )
