# centaur/schema_training_session_results.py
from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- Stores recommendations produced from a training session.
-- Multiple rows over time per session (beginning-of-night + mid-night updates).
CREATE TABLE IF NOT EXISTS training_session_results (
    training_result_id INTEGER PRIMARY KEY,
    training_session_id INTEGER NOT NULL,

    computed_utc TEXT NOT NULL,
    status TEXT NOT NULL,                 -- "ok" | "warn" | "failed"
    message TEXT,

    -- The plan the engine recommends
    recommended_exptime_s REAL,           -- "common exposure length"
    recommended_filters_json TEXT,        -- list of filters considered
    recommended_time_fraction_json TEXT,  -- {Ha:0.25,OIII:0.30,SII:0.45} etc
    recommended_ratio_vs_ha_json TEXT,    -- {Ha:1,OIII:1.1,SII:1.7} etc

    -- Why / context snapshot
    constraints_json TEXT,                -- includes dark-library supported exposures etc
    conditions_json TEXT,                 -- airmass/moon/transparency snapshot
    stats_json TEXT,                      -- E_avg, weights, outliers removed, etc

    FOREIGN KEY (training_session_id)
      REFERENCES training_sessions (training_session_id)
      ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_training_results_session
  ON training_session_results(training_session_id);

CREATE INDEX IF NOT EXISTS idx_training_results_time
  ON training_session_results(computed_utc);
"""


def ensure_training_session_results_schema(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA foreign_keys = ON;")
        con.executescript(SCHEMA_SQL)
        con.commit()
    finally:
        con.close()
