#!/usr/bin/env python3
# centaur/tools/training_session_tag.py
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from typing import Optional

from centaur.logging import utc_now


def _norm_target(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().strip("'\"")
    t = re.sub(r"\s+", " ", t)
    return t.lower() if t else None


def _resolve_latest_open_session(
    con: sqlite3.Connection,
    *,
    target: Optional[str],
    watch_root_id: Optional[int],
) -> Optional[int]:
    where = ["status='open'"]
    params = []

    if target:
        where.append("LOWER(TRIM(target_name)) = ?")
        params.append(_norm_target(target))

    if watch_root_id is not None:
        where.append("watch_root_id = ?")
        params.append(int(watch_root_id))

    row = con.execute(
        f"""
        SELECT training_session_id
        FROM training_sessions
        WHERE {" AND ".join(where)}
        ORDER BY created_utc DESC
        LIMIT 1
        """,
        tuple(params),
    ).fetchone()
    return int(row[0]) if row else None


def _session_exists(con: sqlite3.Connection, sid: int) -> bool:
    row = con.execute(
        "SELECT 1 FROM training_sessions WHERE training_session_id=?",
        (sid,),
    ).fetchone()
    return row is not None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Tag frames into a training session (writes training_session_frames)."
    )
    ap.add_argument("--db", required=True, help="Path to sqlite DB")

    # Session selection
    ap.add_argument(
        "--session", type=int, default=None, help="training_session_id (explicit)"
    )
    ap.add_argument(
        "--session-latest-open",
        action="store_true",
        help="Use the latest open session (optionally filtered by --target/--watch-root-id)",
    )
    ap.add_argument(
        "--target", default=None, help="Target name (used with --session-latest-open)"
    )
    ap.add_argument(
        "--watch-root-id",
        type=int,
        default=None,
        help="watch_root_id filter (multi-rig)",
    )

    ap.add_argument(
        "--tagged-by", default="cli", help='Tag provenance (default: "cli")'
    )
    ap.add_argument("--role", default="train", help='Role label (default: "train")')
    ap.add_argument(
        "--only-usable",
        action="store_true",
        help="Only tag frames where training_derived_metrics.usable=1",
    )

    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--last", type=int, help="Tag the most recent N LIGHT frames")
    mx.add_argument(
        "--since-mins",
        type=int,
        help="Tag LIGHT frames from the last N minutes (by images.db_created_utc)",
    )
    mx.add_argument(
        "--like",
        type=str,
        help='Tag frames where images.file_path LIKE this pattern (e.g. "%Heart Nebula%_Ha_%")',
    )

    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        sid = args.session
        if args.session_latest_open:
            sid = _resolve_latest_open_session(
                con, target=args.target, watch_root_id=args.watch_root_id
            )
            if sid is None:
                print(
                    "ERROR: No open training session found matching the filters.",
                    file=sys.stderr,
                )
                return 2

        if sid is None:
            print(
                "ERROR: Must provide --session or --session-latest-open",
                file=sys.stderr,
            )
            return 2

        if not _session_exists(con, int(sid)):
            print(f"ERROR: training_session_id {sid} not found.", file=sys.stderr)
            return 2

        usable_join = ""
        usable_where = ""
        if args.only_usable:
            usable_join = "JOIN training_derived_metrics td ON td.image_id = i.image_id"
            usable_where = "AND td.usable = 1"

        where = ["COALESCE(UPPER(TRIM(h.imagetyp)),'') = 'LIGHT'"]
        params = []
        order_limit = ""

        if args.watch_root_id is not None:
            where.append("i.watch_root_id = ?")
            params.append(int(args.watch_root_id))

        if args.last is not None:
            order_limit = "ORDER BY i.db_created_utc DESC LIMIT ?"
            params.append(int(args.last))

        elif args.since_mins is not None:
            where.append("i.db_created_utc >= datetime('now', ?)")
            params.append(f"-{int(args.since_mins)} minutes")
            order_limit = "ORDER BY i.db_created_utc DESC"

        elif args.like is not None:
            where.append("i.file_path LIKE ?")
            params.append(args.like)
            order_limit = "ORDER BY i.db_created_utc DESC"

        sql = f"""
        SELECT
            i.image_id,
            i.file_path,
            h.filter AS filter,
            h.exptime AS exptime_s,
            h.gain AS gain_setting
        FROM images i
        JOIN fits_header_core h ON h.image_id = i.image_id
        {usable_join}
        WHERE {" AND ".join(where)}
        {usable_where}
        {order_limit}
        """

        rows = con.execute(sql, tuple(params)).fetchall()
        if not rows:
            print("No matching frames found to tag.")
            return 0

        now = utc_now()
        n_inserted = 0
        n_skipped_dupe = 0

        with con:
            for r in rows:
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
                        int(r["image_id"]),
                        args.role,
                        r["filter"],
                        r["exptime_s"],
                        r["gain_setting"],
                        now,
                        args.tagged_by,
                    ),
                )
                if cur.rowcount == 1:
                    n_inserted += 1
                else:
                    n_skipped_dupe += 1

            con.execute(
                "UPDATE training_sessions SET db_updated_utc=? WHERE training_session_id=?",
                (now, int(sid)),
            )

        print("Training session tagging complete")
        print("================================")
        print(f"DB: {args.db}")
        print(f"training_session_id: {sid}")
        print(f"tagged_by: {args.tagged_by}")
        print(f"role: {args.role}")
        print(f"only_usable: {bool(args.only_usable)}")
        if args.watch_root_id is not None:
            print(f"watch_root_id: {int(args.watch_root_id)}")
        print(f"rows_found: {len(rows)}")
        print(f"inserted: {n_inserted}")
        print(f"already_tagged: {n_skipped_dupe}")
        return 0

    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
