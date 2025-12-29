#!/usr/bin/env python3
# centaur/model_tests/models/test_training_derived_metrics.py
#
# Model test for training_derived_metrics.
# This is a support table used by training_session_engine.
#
# FIX (2025-12-28):
# - Allow training_derived_metrics.filter to be blank ONLY for OSC frames.
# - OSC detection: presence of BAYERPAT keyword (non-empty value) in fits_header_full.header_json.
# - For non-OSC frames, filter remains REQUIRED (non-empty).
#
# Exit codes:
#   0 PASS
#   2 FAIL
#   1 ERROR

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "training_derived_metrics"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return bool(r)


def _cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [
        str(r["name"]) for r in conn.execute(f"PRAGMA table_info({table});").fetchall()
    ]


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in set(_cols(conn, table))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        if v in (float("inf"), float("-inf")):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _truthy_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


def _resolve_output_dir(args_run_dir: str, stamp: str) -> Path:
    if args_run_dir.strip():
        out_dir = Path(args_run_dir) / MODEL_NAME
    else:
        out_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@dataclass
class FailRow:
    image_id: int
    issues: str


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not _table_exists(conn, MODEL_NAME):
        return False, [f"missing_table:{MODEL_NAME}"]

    # We use fits_header_full to detect OSC (BAYERPAT present)
    if not _table_exists(conn, "fits_header_full"):
        issues.append("missing_table:fits_header_full")

    for col in (
        "image_id",
        "usable",
        "exptime_s",
        "filter",
        "nebula_minus_bg_adu_s",
        "sky_ff_median_adu_s",
    ):
        if not _has_col(conn, MODEL_NAME, col):
            issues.append(f"{MODEL_NAME}_missing_col:{col}")

    return (len(issues) == 0), issues


def _load_has_bayerpat(
    conn: sqlite3.Connection, image_ids: List[int]
) -> Dict[int, int]:
    """
    Returns dict: image_id -> 1 if BAYERPAT keyword exists (non-empty value), else 0.
    Chunked to avoid SQLite parameter limits.
    """
    out: Dict[int, int] = {}
    if not image_ids:
        return out
    if not _table_exists(conn, "fits_header_full"):
        return out

    CHUNK = 900
    for i0 in range(0, len(image_ids), CHUNK):
        chunk = image_ids[i0 : i0 + CHUNK]
        qmarks = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT
              ff.image_id AS image_id,
              CASE WHEN EXISTS (
                SELECT 1
                FROM json_each(ff.header_json) je
                WHERE UPPER(TRIM(json_extract(je.value,'$.keyword')))='BAYERPAT'
                  AND TRIM(COALESCE(json_extract(je.value,'$.value'),'')) <> ''
              ) THEN 1 ELSE 0 END AS has_bayerpat
            FROM fits_header_full ff
            WHERE ff.image_id IN ({qmarks})
            """,
            tuple(chunk),
        ).fetchall()

        for r in rows:
            out[int(r["image_id"])] = int(r["has_bayerpat"])

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {MODEL_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run dir; outputs go into <run-dir>/<model>/",
    )
    ap.add_argument(
        "--out", type=str, default="", help="Output JSON path (optional override)"
    )
    ap.add_argument(
        "--csv", type=str, default="", help="Failures CSV path (optional override)"
    )
    ap.add_argument(
        "--max-fail-rows", type=int, default=300, help="Max failing rows to capture"
    )
    args = ap.parse_args()

    stamp = _utc_stamp()
    out_dir = _resolve_output_dir(args.run_dir, stamp)
    out_json = (
        Path(args.out)
        if args.out.strip()
        else (out_dir / f"test_{MODEL_NAME}_{stamp}.json")
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else (out_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv")
    )

    try:
        conn = _connect(Path(args.db))
    except Exception as e:
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        return 1

    ok_schema, schema_issues = _required_schema_checks(conn)
    if not ok_schema:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(args.db),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    colset = set(_cols(conn, MODEL_NAME))
    has_lh = "linear_headroom_p99" in colset
    has_slr = "sky_limited_ratio" in colset
    has_tp = "transparency_proxy" in colset
    has_nos = "nebula_over_sky" in colset

    rows = [
        dict(r)
        for r in conn.execute(
            f"SELECT * FROM {MODEL_NAME} ORDER BY image_id;"
        ).fetchall()
    ]
    n_total = len(rows)

    # Build OSC classifier: BAYERPAT present => OSC
    image_ids = [
        int(r["image_id"]) for r in rows if _safe_int(r.get("image_id")) is not None
    ]
    has_bayerpat = _load_has_bayerpat(conn, image_ids)

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []
    fail_reasons: List[str] = []

    # Uniqueness: one row per image_id
    dup = conn.execute(
        f"SELECT COUNT(*) AS n_bad FROM (SELECT image_id FROM {MODEL_NAME} GROUP BY 1 HAVING COUNT(*) > 1)"
    ).fetchone()
    if dup and int(dup["n_bad"]) > 0:
        issue_counts["duplicate_image_id_rows"] = int(dup["n_bad"])

    for r in rows:
        issues: List[str] = []
        image_id = _safe_int(r.get("image_id")) or 0

        u = _safe_int(r.get("usable"))
        if u is None:
            _add_issue(issues, "usable_missing")
        elif u not in (0, 1):
            _add_issue(issues, "usable_not_0_or_1")

        exptime_s = _safe_float(r.get("exptime_s"))
        if exptime_s is not None and exptime_s <= 0:
            _add_issue(issues, "exptime_s_nonpos")

        # FILTER RULE:
        # - Mono/non-OSC => filter required
        # - OSC (BAYERPAT present) => filter may be blank
        filt = r.get("filter")
        if not _truthy_str(filt):
            is_osc = has_bayerpat.get(int(image_id), 0) == 1
            if not is_osc:
                _add_issue(issues, "filter_missing_or_empty_non_osc")

        neb = _safe_float(r.get("nebula_minus_bg_adu_s"))
        sky = _safe_float(r.get("sky_ff_median_adu_s"))

        if u == 1:
            if neb is None:
                _add_issue(issues, "usable_but_nebula_minus_bg_missing")
            elif neb < 0:
                _add_issue(issues, "nebula_minus_bg_negative")
            if sky is None:
                _add_issue(issues, "usable_but_sky_ff_median_missing")
            elif sky <= 0:
                _add_issue(issues, "sky_ff_median_nonpos")

        # Optional sanity (only if cols exist)
        if has_lh:
            lh = _safe_float(r.get("linear_headroom_p99"))
            if lh is not None and not (-10.0 <= lh <= 10.0):
                _add_issue(issues, "linear_headroom_p99_extreme")

        if has_slr:
            slr = _safe_float(r.get("sky_limited_ratio"))
            if slr is not None and slr < 0:
                _add_issue(issues, "sky_limited_ratio_negative")

        if has_tp:
            tp = _safe_float(r.get("transparency_proxy"))
            if tp is not None and tp <= 0:
                _add_issue(issues, "transparency_proxy_nonpos")

        if has_nos:
            nos = _safe_float(r.get("nebula_over_sky"))
            if nos is not None and nos <= 0:
                _add_issue(issues, "nebula_over_sky_nonpos")

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1
            if len(failing) < int(args.max_fail_rows):
                failing.append(FailRow(image_id=int(image_id), issues=";".join(issues)))

    if n_total == 0:
        fail_reasons.append("no_rows_in_training_derived_metrics")

    if issue_counts:
        fail_reasons.append(f"invariant_violations({sum(issue_counts.values())})")

    status = "PASS" if not fail_reasons else "FAIL"
    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(args.db),
        "population": {"n_rows": n_total},
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "fail_reasons": fail_reasons,
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "optional_cols_seen": {
                "linear_headroom_p99": has_lh,
                "sky_limited_ratio": has_slr,
                "transparency_proxy": has_tp,
                "nebula_over_sky": has_nos,
            },
            "osc_detection": "BAYERPAT present in fits_header_full.header_json => OSC; OSC may have blank filter",
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "issues"])
            for fr in failing:
                w.writerow([fr.image_id, fr.issues])

    print(f"[test_{MODEL_NAME}] status={status} n_rows={n_total}")
    if fail_reasons:
        for rr in fail_reasons:
            print(f"  reason: {rr}")
    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  issue::{k}={v}")

    print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
    if write_csv:
        print(f"[test_{MODEL_NAME}] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
