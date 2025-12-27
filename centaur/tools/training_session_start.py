#!/usr/bin/env python3
# centaur/tools/training_session_start.py
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from typing import Any, Dict, Optional, Tuple

from centaur.config import default_config
from centaur.logging import utc_now


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


def _fetch_watch_roots(con: sqlite3.Connection) -> list[Tuple[int, str, Optional[str]]]:
    if not _table_exists(con, "watch_roots"):
        return []
    rows = con.execute(
        "SELECT watch_root_id, root_path, root_label FROM watch_roots ORDER BY watch_root_id"
    ).fetchall()
    out: list[Tuple[int, str, Optional[str]]] = []
    for r in rows:
        out.append((int(r[0]), str(r[1]), (str(r[2]) if r[2] is not None else None)))
    return out


def _watch_root_exists(con: sqlite3.Connection, watch_root_id: int) -> bool:
    if not _table_exists(con, "watch_roots"):
        return False
    row = con.execute(
        "SELECT 1 FROM watch_roots WHERE watch_root_id=?",
        (int(watch_root_id),),
    ).fetchone()
    return row is not None


def _prompt_str(
    label: str, default: Optional[str] = None, *, required: bool = False
) -> str:
    while True:
        if default is not None and default != "":
            s = input(f"{label} [{default}]: ").strip()
            if s == "":
                s = str(default).strip()
        else:
            s = input(f"{label}: ").strip()

        if required and not s:
            print("  -> required (cannot be blank)")
            continue
        return s


def _prompt_int(label: str, default: int) -> int:
    while True:
        s = input(f"{label} [{default}]: ").strip()
        if s == "":
            return int(default)
        try:
            return int(s)
        except Exception:
            print("  -> please enter an integer")


def _prompt_float(label: str, default: float) -> float:
    while True:
        s = input(f"{label} [{default}]: ").strip()
        if s == "":
            return float(default)
        try:
            return float(s)
        except Exception:
            print("  -> please enter a number")


def _parse_filters(s: str) -> Optional[list[str]]:
    """
    Accepts:
      - "HSO", "H,S,O", "H O S"
      - "HA,OIII,SII"
      - mixed case
    Returns canonical list like ["HA","SII","OIII"] in the entered order (deduped).
    """
    if s is None:
        return None
    raw = s.strip().upper()
    if not raw:
        return None

    # Turn separators into spaces
    raw = re.sub(r"[,\|;/]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    tokens: list[str] = []
    # If user typed "HSO" with no spaces, split to characters when it looks like shorthand
    if " " not in raw and all(ch in "HSO" for ch in raw) and len(raw) <= 3:
        tokens = list(raw)
    else:
        tokens = raw.split(" ")

    mapping = {
        "H": "HA",
        "HA": "HA",
        "O": "OIII",
        "OIII": "OIII",
        "S": "SII",
        "SII": "SII",
    }

    out: list[str] = []
    for t in tokens:
        t = t.strip().upper()
        if not t:
            continue
        if t not in mapping:
            return None
        canon = mapping[t]
        if canon not in out:
            out.append(canon)

    return out if out else None


def _prompt_filters(label: str, default_filters: list[str]) -> list[str]:
    default_str = "".join(
        [{"HA": "H", "SII": "S", "OIII": "O"}.get(f, f) for f in default_filters]
    )
    while True:
        s = input(f"{label} [{default_str}]: ").strip()
        if s == "":
            return list(default_filters)
        parsed = _parse_filters(s)
        if parsed:
            return parsed
        print(
            "  -> invalid filters. Use H/S/O (e.g. HSO or H,S,O) or full names (HA SII OIII)."
        )


def _prompt_watch_root_id(
    con: sqlite3.Connection, default: Optional[int]
) -> Optional[int]:
    roots = _fetch_watch_roots(con)

    print("\nWatch roots (optional)")
    print("----------------------")
    if not roots:
        print("  (none found in DB yet)")
        print("  Tip: run cp_start once so the watcher inserts watch_roots rows.\n")
        s = input("watch_root_id (blank for none): ").strip()
        if s == "":
            return None
        try:
            wid = int(s)
            if _watch_root_exists(con, wid):
                return wid
        except Exception:
            pass
        print("  -> invalid/unknown watch_root_id. Using none.\n")
        return None

    for wid, path, label in roots:
        lbl = f" ({label})" if label else ""
        print(f"  {wid}: {path}{lbl}")

    while True:
        if default is None:
            s = input("watch_root_id (blank for none): ").strip()
        else:
            s = input(f"watch_root_id (blank for none) [{default}]: ").strip()
            if s == "":
                if _watch_root_exists(con, int(default)):
                    return int(default)
                print(
                    f"  -> watch_root_id {default} not found. Choose from list or blank."
                )
                continue

        if s == "":
            return None

        try:
            wid = int(s)
        except Exception:
            print("  -> please enter an integer id from the list, or blank for none.")
            continue

        if _watch_root_exists(con, wid):
            return wid

        print(f"  -> watch_root_id {wid} not found. Choose from list or blank.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a new training session (open). Interactive by default; args become defaults."
    )
    ap.add_argument(
        "--db", required=True, help="Path to sqlite DB (e.g. data/centaurparting.db)"
    )
    ap.add_argument(
        "--target", default=None, help="Target name (required; prompted if missing)"
    )
    ap.add_argument(
        "--watch-root-id", type=int, default=None, help="Optional watch_root_id"
    )
    ap.add_argument(
        "--finish-after",
        type=int,
        default=None,
        help="Auto-close after N tagged frames (0 = never)",
    )
    ap.add_argument("--label", default=None, help="Optional session label")
    ap.add_argument("--notes", default=None, help="Optional notes")
    ap.add_argument(
        "--params-json",
        default=None,
        help="Optional JSON object string. Interactive inputs will override/augment it.",
    )
    args = ap.parse_args()

    cfg = default_config()

    params: Dict[str, Any] = {}
    if args.params_json:
        try:
            params = json.loads(args.params_json)
            if not isinstance(params, dict):
                print(
                    'ERROR: --params-json must be a JSON object like {"a":1}',
                    file=sys.stderr,
                )
                return 2
        except Exception as e:
            print(f"ERROR: --params-json is not valid JSON: {e}", file=sys.stderr)
            return 2

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        print("CentaurParting - Training Session Start")
        print("======================================")

        target_default = _norm_target(args.target)
        target = _prompt_str("Target name", target_default, required=True)
        target_norm = _norm_target(target)
        if not target_norm:
            print("ERROR: target name cannot be blank", file=sys.stderr)
            return 2

        watch_root_id = _prompt_watch_root_id(con, args.watch_root_id)

        finish_after_default = (
            0 if args.finish_after is None else int(args.finish_after)
        )
        finish_after = _prompt_int(
            "Finish after N frames? (0 = no auto-finish)", finish_after_default
        )
        if finish_after < 0:
            finish_after = 0

        label_default = (args.label or "").strip() or None
        label = (
            _prompt_str(
                "Session label (optional)", label_default, required=False
            ).strip()
            or None
        )

        notes_default = (args.notes or "").strip() or None
        notes = (
            _prompt_str("Notes (optional)", notes_default, required=False).strip()
            or None
        )

        # Recommendation defaults from global config (Enter=accept defaults)
        print("\nRecommendation defaults (Enter = accept)")
        print("--------------------------------------")

        req_filters = _prompt_filters(
            "Required filters (H,S,O shorthand allowed)",
            list(
                getattr(cfg, "training_required_filters_default", ["SII", "HA", "OIII"])
            ),
        )
        min_lh = _prompt_float(
            "Min linear headroom p99 (e.g. 0.10)",
            float(getattr(cfg, "training_min_linear_headroom_p99", 0.10)),
        )
        w_mean = _prompt_float(
            "Score weight: mean (w_mean)",
            float(getattr(cfg, "training_score_w_mean", 0.55)),
        )
        w_min = _prompt_float(
            "Score weight: min  (w_min)",
            float(getattr(cfg, "training_score_w_min", 0.45)),
        )
        short_pen = _prompt_float(
            "Penalty: short exposures",
            float(getattr(cfg, "training_short_penalty", 0.10)),
        )
        long_pen = _prompt_float(
            "Penalty: long exposures",
            float(getattr(cfg, "training_long_penalty", 0.08)),
        )

        # final FK guard
        if watch_root_id is not None and not _watch_root_exists(
            con, int(watch_root_id)
        ):
            print(
                f"ERROR: watch_root_id {watch_root_id} not found in watch_roots. Use blank for none.",
                file=sys.stderr,
            )
            return 2

        # Write session params: defaults + overrides live here
        params = dict(params)
        params["finish_after"] = int(finish_after)
        params["required_filters"] = req_filters
        params["min_linear_headroom_p99"] = float(min_lh)
        params["w_mean"] = float(w_mean)
        params["w_min"] = float(w_min)
        params["short_penalty"] = float(short_pen)
        params["long_penalty"] = float(long_pen)

        now = utc_now()
        cur = con.execute(
            """
            INSERT INTO training_sessions (
                session_label,
                target_name,
                watch_root_id,
                created_utc,
                started_utc,
                ended_utc,
                status,
                notes,
                params_json,
                results_json,
                db_updated_utc
            ) VALUES (?, ?, ?, ?, ?, NULL, 'open', ?, ?, NULL, ?)
            """,
            (
                label,
                target_norm,
                watch_root_id,
                now,
                now,
                notes,
                json.dumps(params),
                now,
            ),
        )
        con.commit()
        sid = int(cur.lastrowid)

        print("\nTraining session created")
        print("=======================")
        print(f"DB: {args.db}")
        print(f"training_session_id: {sid}")
        print("status: open")
        print(f"target_name: {target_norm}")
        print(f"watch_root_id: {watch_root_id if watch_root_id is not None else ''}")
        print(f"finish_after: {finish_after}")
        if label:
            print(f"session_label: {label}")

        print("\nNEXT:")
        print("  1) Run: cp_start --training-monitor")
        print("  2) Drop training frames into the watcher root")
        print("  3) Auto-tagging + auto-close will happen in the cp_start log")
        return 0

    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
