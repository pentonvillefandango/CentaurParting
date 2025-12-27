# centaur/training_session_monitor.py
from __future__ import annotations

import json
import sqlite3
import threading
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple, List

from centaur.logging import Logger, utc_now


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


def _safe_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


@dataclass
class SessionState:
    last_seen_image_id: int = 0
    near_miss_seen: Dict[str, float] = field(default_factory=dict)
    last_progress: int = 0


class TrainingSessionMonitor:
    """
    Lightweight coordinator:
      - Finds open training_sessions
      - Auto-tags matching frames into training_session_frames
      - Logs progress (n/finish_after)
      - Auto-solves + closes when finish_after reached (writes training_session_results)
    """

    def __init__(
        self,
        *,
        db_path: str,
        logger: Logger,
        poll_seconds: float = 5.0,
        only_usable: bool = True,
    ) -> None:
        self._db_path = db_path
        self._logger = logger
        self._poll_seconds = max(0.5, float(poll_seconds))
        self._only_usable = bool(only_usable)
        self._stop = threading.Event()
        self._states: Dict[int, SessionState] = {}

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        self._logger.log_module_summary(
            "training_session_monitor",
            "(startup)",
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="OK",
            duration_s=0.0,
        )

        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                self._logger.log_failure(
                    "training_session_monitor",
                    "(tick)",
                    action="continue",
                    reason=repr(e),
                )
            self._stop.wait(self._poll_seconds)

        self._logger.log_module_summary(
            "training_session_monitor",
            "(shutdown)",
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="Stopped",
            duration_s=0.0,
        )

    def _tick(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys = ON;")

            if not _table_exists(con, "training_sessions") or not _table_exists(
                con, "training_session_frames"
            ):
                return

            open_rows = con.execute(
                """
                SELECT training_session_id, target_name, watch_root_id, params_json
                FROM training_sessions
                WHERE status='open'
                ORDER BY training_session_id
                """
            ).fetchall()

            if not open_rows:
                return

            open_ids = {int(r["training_session_id"]) for r in open_rows}
            for sid in list(self._states.keys()):
                if sid not in open_ids:
                    self._states.pop(sid, None)

            for r in open_rows:
                sid = int(r["training_session_id"])
                target = _norm_target(r["target_name"])
                watch_root_id = _safe_int(r["watch_root_id"])
                params_json = r["params_json"]

                params = {}
                if params_json:
                    try:
                        params = json.loads(params_json)
                        if not isinstance(params, dict):
                            params = {}
                    except Exception:
                        params = {}

                finish_after = _safe_int(params.get("finish_after")) or 0

                self._process_one_session(
                    con,
                    sid=sid,
                    target=target,
                    watch_root_id=watch_root_id,
                    finish_after=finish_after,
                )

        finally:
            con.close()

    def _process_one_session(
        self,
        con: sqlite3.Connection,
        *,
        sid: int,
        target: Optional[str],
        watch_root_id: Optional[int],
        finish_after: int,
    ) -> None:
        st = self._states.setdefault(sid, SessionState())

        tagged_now = con.execute(
            "SELECT COUNT(*) FROM training_session_frames WHERE training_session_id=?",
            (sid,),
        ).fetchone()[0]
        tagged_now = int(tagged_now or 0)

        if finish_after > 0 and tagged_now >= finish_after:
            if st.last_progress < finish_after:
                st.last_progress = tagged_now
                self._log_progress(
                    sid, tagged_now, finish_after, note="(already reached)"
                )
            self._solve_and_close(sid)
            return

        usable_join = ""
        usable_where = ""
        if self._only_usable:
            usable_join = "JOIN training_derived_metrics td ON td.image_id = i.image_id"
            usable_where = "AND td.usable = 1"

        where = """
        WHERE i.image_id > ?
          AND COALESCE(UPPER(TRIM(h.imagetyp)),'') = 'LIGHT'
        """
        params: List[object] = [int(st.last_seen_image_id)]

        if watch_root_id is not None:
            where += " AND i.watch_root_id = ?"
            params.append(int(watch_root_id))

        if target:
            where += " AND h.object IS NOT NULL"
        else:
            return

        sql = f"""
        SELECT
            i.image_id,
            i.file_path,
            h.object AS object_name,
            h.filter AS filter,
            h.exptime AS exptime_s,
            h.gain AS gain_setting
        FROM images i
        JOIN fits_header_core h ON h.image_id = i.image_id
        {usable_join}
        {where}
        {usable_where}
        ORDER BY i.image_id ASC
        LIMIT 200
        """

        rows = con.execute(sql, tuple(params)).fetchall()
        if not rows:
            return

        st.last_seen_image_id = int(rows[-1]["image_id"])

        n_tagged = 0
        now = utc_now()
        with con:
            for row in rows:
                image_id = int(row["image_id"])
                obj = _norm_target(row["object_name"])
                if obj != target:
                    if obj and obj not in st.near_miss_seen:
                        sim = _similarity(obj, target)
                        if sim >= 0.82:
                            st.near_miss_seen[obj] = sim
                            self._logger.log_module_result(
                                "training_session_monitor",
                                str(row["file_path"]),
                                expected_read=1,
                                read=1,
                                expected_written=1,
                                written=0,
                                status="SKIPPED",
                                duration_s=0.0,
                                verbose_fields={
                                    "__outputs__": {
                                        "reason": "near_miss_target",
                                        "session_id": sid,
                                        "session_target": target,
                                        "image_target": obj,
                                        "similarity": f"{sim:.2f}",
                                    }
                                },
                            )
                    continue

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
                    ) VALUES (?, ?, 'train', ?, ?, ?, ?, 'monitor')
                    """,
                    (
                        int(sid),
                        int(image_id),
                        row["filter"],
                        row["exptime_s"],
                        row["gain_setting"],
                        now,
                    ),
                )
                if cur.rowcount == 1:
                    n_tagged += 1

            con.execute(
                "UPDATE training_sessions SET db_updated_utc=? WHERE training_session_id=?",
                (now, int(sid)),
            )

        if n_tagged > 0:
            tagged_now2 = con.execute(
                "SELECT COUNT(*) FROM training_session_frames WHERE training_session_id=?",
                (sid,),
            ).fetchone()[0]
            tagged_now2 = int(tagged_now2 or 0)

            if finish_after > 0:
                self._log_progress(sid, tagged_now2, finish_after)

            if finish_after > 0 and tagged_now2 >= finish_after:
                self._solve_and_close(sid)

    def _log_progress(
        self, sid: int, tagged_total: int, finish_after: int, note: str = ""
    ) -> None:
        st = self._states.setdefault(sid, SessionState())
        if tagged_total <= st.last_progress:
            return
        st.last_progress = tagged_total

        msg = f"{tagged_total}/{finish_after}"
        if note:
            msg += f" {note}"

        self._logger.log_module_result(
            "training_session_monitor",
            f"(session {sid})",
            expected_read=1,
            read=1,
            expected_written=1,
            written=1,
            status="OK",
            duration_s=0.0,
            verbose_fields={"__outputs__": {"progress": msg}},
        )

    def _solve_and_close(self, sid: int) -> None:
        """
        Solve + close within this process.
        IMPORTANT (Option A):
          - log ONLY the human report (no ugly dict)
          - structured detail remains in DB (training_session_results.stats_json)
        """
        from centaur.tools.training_session_close import (
            solve_session_and_optionally_close,
        )

        ok, summary = solve_session_and_optionally_close(
            db_path=self._db_path,
            session_id=sid,
            close_session=True,
        )

        if ok:
            report_text = str(summary.get("report_text") or "").strip()
            if not report_text:
                report_text = "TRAINING RECOMMENDATION (no report_text returned)"

            self._logger.log_module_result(
                "training_session_monitor",
                f"(session {sid})",
                expected_read=1,
                read=1,
                expected_written=1,
                written=1,
                status="OK",
                duration_s=0.0,
                verbose_fields={"__outputs__": {"report": report_text}},
            )
        else:
            self._logger.log_failure(
                "training_session_monitor",
                f"(session {sid})",
                action="continue",
                reason=str(summary.get("error") or "solve_failed"),
            )


def start_monitor_thread(
    *,
    db_path: str,
    logger: Logger,
    poll_seconds: float = 5.0,
    only_usable: bool = True,
) -> Tuple[threading.Thread, TrainingSessionMonitor]:
    mon = TrainingSessionMonitor(
        db_path=str(db_path),
        logger=logger,
        poll_seconds=float(poll_seconds),
        only_usable=bool(only_usable),
    )
    t = threading.Thread(
        target=mon.run_forever, name="training_session_monitor", daemon=True
    )
    t.start()
    return t, mon
