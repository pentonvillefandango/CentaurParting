# centaur/training_session_tracker.py
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from centaur.logging import Logger, utc_now
from centaur.training_session_engine import solve_training_session, DEFAULT_FILTERS

MODULE_NAME = "training_session_tracker"


def _norm_target(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = t.strip("'\"")
    t = re.sub(r"\s+", " ", t)
    return t.lower()


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _get_image_identity(
    con: sqlite3.Connection, file_path: str
) -> Optional[Tuple[int, Optional[int]]]:
    row = con.execute(
        "SELECT image_id, watch_root_id FROM images WHERE file_path=?",
        (file_path,),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), (_safe_int(row[1]))


def _get_header_object_and_type(
    con: sqlite3.Connection, image_id: int
) -> Tuple[str, str]:
    row = con.execute(
        "SELECT object, imagetyp FROM fits_header_core WHERE image_id=?",
        (image_id,),
    ).fetchone()
    if row is None:
        return "", ""
    return str(row[0] or ""), str(row[1] or "")


def _session_params(con: sqlite3.Connection, sid: int) -> Dict[str, Any]:
    row = con.execute(
        "SELECT params_json FROM training_sessions WHERE training_session_id=?",
        (int(sid),),
    ).fetchone()
    if row is None or row[0] is None:
        return {}
    try:
        v = json.loads(row[0])
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _count_tagged_frames(con: sqlite3.Connection, sid: int) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM training_session_frames WHERE training_session_id=?",
        (int(sid),),
    ).fetchone()
    return int(row[0] or 0)


def _tag_frame(
    con: sqlite3.Connection,
    *,
    sid: int,
    image_id: int,
    tagged_by: str,
    role: str,
) -> bool:
    row = con.execute(
        "SELECT filter, exptime, gain FROM fits_header_core WHERE image_id=?",
        (image_id,),
    ).fetchone()
    filt = row[0] if row else None
    exptime_s = row[1] if row else None
    gain_setting = row[2] if row else None

    now = utc_now()
    cur = con.execute(
        """
        INSERT OR IGNORE INTO training_session_frames (
            training_session_id,
            image_id,
            role,
            filter,
            exptime_s,
            gain_setting,
            tagged_utc,
            tagged_by
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(sid),
            int(image_id),
            role,
            filt,
            exptime_s,
            gain_setting,
            now,
            tagged_by,
        ),
    )
    return cur.rowcount == 1


def handle_new_file_ingested(db_path: Path, logger: Logger, file_path: str) -> None:
    """
    Legacy hook (if still used): auto-tag into open sessions and auto-close when finish_after is reached.

    IMPORTANT:
    - Uses centaur.training_session_engine.solve_training_session() (single source of truth).
    - Logs a compact one-liner only.
    - Full detail is stored in training_session_results.stats_json in the DB.
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        if not _table_exists(con, "training_sessions") or not _table_exists(
            con, "training_session_frames"
        ):
            return

        ident = _get_image_identity(con, file_path)
        if ident is None:
            return
        image_id, image_watch_root_id = ident

        obj, imagetyp = _get_header_object_and_type(con, image_id)
        if str(imagetyp or "").strip().upper() != "LIGHT":
            return

        obj_norm = _norm_target(obj)
        if not obj_norm:
            return

        sessions = con.execute(
            """
            SELECT training_session_id, target_name, watch_root_id
            FROM training_sessions
            WHERE status='open'
              AND target_name IS NOT NULL
            ORDER BY training_session_id
            """
        ).fetchall()

        for s in sessions:
            sid = int(s["training_session_id"])
            session_target_norm = _norm_target(s["target_name"])
            if session_target_norm != obj_norm:
                continue

            sess_watch_root_id = _safe_int(s["watch_root_id"])
            if sess_watch_root_id is not None and image_watch_root_id is not None:
                if int(sess_watch_root_id) != int(image_watch_root_id):
                    continue
            elif sess_watch_root_id is not None and image_watch_root_id is None:
                continue

            with con:
                inserted = _tag_frame(
                    con, sid=sid, image_id=image_id, tagged_by="auto", role="train"
                )
                if inserted:
                    now = utc_now()
                    con.execute(
                        "UPDATE training_sessions SET db_updated_utc=? WHERE training_session_id=?",
                        (now, int(sid)),
                    )

            if inserted:
                logger.log_module_result(
                    MODULE_NAME,
                    file_path,
                    expected_read=1,
                    read=1,
                    expected_written=1,
                    written=1,
                    status="OK",
                    duration_s=None,
                    verbose_fields={
                        "__inputs__": {"target": obj_norm, "training_session_id": sid},
                        "__outputs__": {"tagged_image_id": image_id},
                    },
                )

            params = _session_params(con, sid)
            finish_after = _safe_int(params.get("finish_after"))
            if finish_after is None or finish_after <= 0:
                continue

            n_tagged = _count_tagged_frames(con, sid)
            if n_tagged < int(finish_after):
                continue

            # Close and solve once
            now = utc_now()
            with con:
                con.execute(
                    """
                    UPDATE training_sessions
                    SET status='closed', ended_utc=?, db_updated_utc=?
                    WHERE training_session_id=? AND status='open'
                    """,
                    (now, now, int(sid)),
                )

                result = solve_training_session(
                    con,
                    sid,
                    required_filters_default=DEFAULT_FILTERS,
                    min_linear_headroom_default=0.02,
                    w_mean_default=0.55,
                    w_min_default=0.45,
                    short_penalty_default=0.10,
                    long_penalty_default=0.08,
                    include_excluded_default=True,
                    close_call_delta_default=0.05,
                )

            if result.get("ok") and result.get("best"):
                best = result["best"]
                rx = float(best.get("recommended_exptime_s") or 0.0)
                ratios = best.get("ratio_vs_ha") or {}
                one_liner = f"{rx:.0f}s | " + " ".join(
                    [f"{k}:{float(v):.2f}x" for k, v in ratios.items()]
                )

                logger.log_module_result(
                    "training_session_close(auto)",
                    file_path,
                    expected_read=1,
                    read=1,
                    expected_written=1,
                    written=1,
                    status="OK",
                    duration_s=None,
                    verbose_fields={
                        "__outputs__": {
                            "session_id": sid,
                            "target_name": obj_norm,
                            "recommendation": one_liner,
                        }
                    },
                )
            else:
                logger.log_failure(
                    "training_session_close(auto)",
                    file_path,
                    action="continue",
                    reason=str(result.get("message") or "failed"),
                    duration_s=None,
                )

    finally:
        con.close()
