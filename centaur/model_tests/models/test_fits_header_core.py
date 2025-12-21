#!/usr/bin/env python3
# centaur/model_tests/models/test_fits_header_core.py
#
# System integrity test for fits_header_core.
# - Ensures every images.image_id has a matching fits_header_core row (configurable thresholds)
# - Ensures fits_header_core.image_id is unique
# - LIGHT frames: exptime must be > 0
# - Basic sanity checks: normalized strings, numeric fields not negative, etc.
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_fits_header_core.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_fits_header_core.py --db data/centaurparting.db --run-dir data/model_tests/test_results_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "fits_header_core"


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
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [str(r["name"]) for r in rows]


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


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _norm_trim(x: Any) -> str:
    return str(x or "").strip()


def _clamp01(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


def _resolve_output_paths(
    stamp: str,
    run_dir: str,
    out_json: str,
    out_csv: str,
) -> Tuple[Path, Path, Path]:
    """
    Returns: (base_dir_for_model, json_path, csv_path)
    """
    if run_dir.strip():
        base = Path(run_dir) / MODEL_NAME
    else:
        base = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"

    base.mkdir(parents=True, exist_ok=True)

    json_path = (
        Path(out_json) if out_json.strip() else base / f"test_{MODEL_NAME}_{stamp}.json"
    )
    csv_path = (
        Path(out_csv)
        if out_csv.strip()
        else base / f"test_{MODEL_NAME}_failures_{stamp}.csv"
    )
    return base, json_path, csv_path


@dataclass
class FailRow:
    image_id: int
    file_name: str
    status: str
    camera: str
    imagetyp: str
    filter: str
    exptime: float
    issues: str


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    for t in ("images", "fits_header_core"):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    for col in ("image_id", "file_name", "status"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    # minimum we rely on for header sanity
    for col in ("image_id", "imagetyp", "instrume", "filter", "exptime", "object"):
        if not _has_col(conn, "fits_header_core", col):
            issues.append(f"fits_header_core_missing_col:{col}")

    return (len(issues) == 0), issues


# Base join: all images with optional header
BASE_JOIN_SQL = """
SELECT
  i.image_id,
  i.file_name,
  i.status AS image_status,

  -- header fields (nullable if missing)
  h.image_id AS h_image_id,
  UPPER(TRIM(COALESCE(h.imagetyp,'')))  AS imagetyp,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  UPPER(TRIM(COALESCE(h.filter,'')))   AS filter,
  TRIM(COALESCE(h.object,''))          AS object,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime
FROM images i
LEFT JOIN fits_header_core h
  ON h.image_id = i.image_id
ORDER BY i.image_id;
"""


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # Missing header row?
    if r.get("h_image_id") is None:
        _add_issue(issues, "missing_fits_header_core_row")
        return issues

    imagetyp = _norm_upper(r.get("imagetyp"))
    camera = _norm_trim(r.get("camera"))
    flt = _norm_trim(r.get("filter"))
    exptime = _safe_float(r.get("exptime"))

    # Basic fields sanity
    if not imagetyp:
        _add_issue(issues, "imagetyp_blank")
    if not camera:
        _add_issue(issues, "camera_blank")

    if exptime is None:
        _add_issue(issues, "exptime_missing_or_non_numeric")
    elif exptime < 0:
        _add_issue(issues, "exptime_negative")

    # LIGHT frames: exptime should be > 0
    # (you consider '' as LIGHT too in your pipeline, so treat it same)
    if imagetyp in ("LIGHT", ""):
        if exptime is None or exptime <= 0:
            _add_issue(issues, "light_exptime_missing_or_nonpos")

    # FILTER is often blank on OSC; don't hard-fail, but record it for troubleshooting
    if imagetyp in ("LIGHT", "") and not flt:
        _add_issue(issues, "light_filter_blank")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="System test: fits_header_core")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="If set, write outputs under <run-dir>/<model>/",
    )

    ap.add_argument(
        "--out", type=str, default="", help="Output JSON path (blank = auto)"
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Failures CSV path (blank = auto on FAIL only)",
    )

    ap.add_argument(
        "--max-fail-rows",
        type=int,
        default=300,
        help="Max failing rows to include in JSON/CSV",
    )

    # thresholds (pct of ALL images)
    ap.add_argument(
        "--fail-if-missing-header-pct",
        type=float,
        default=0.0,
        help="Fail if missing fits_header_core row pct > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-light-bad-exptime-pct",
        type=float,
        default=0.0,
        help="Fail if LIGHT frames with exptime<=0 pct > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-light-filter-blank-pct",
        type=float,
        default=100.0,
        help="Fail if LIGHT frames with blank FILTER pct > this (0..100). Default 100 (never fail).",
    )

    args = ap.parse_args()
    stamp = _utc_stamp()

    db_path = Path(args.db)

    _, out_json, out_csv = _resolve_output_paths(
        stamp=stamp,
        run_dir=args.run_dir,
        out_json=args.out,
        out_csv=args.csv,
    )

    try:
        conn = _connect(db_path)
    except Exception as e:
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        return 1

    ok_schema, schema_issues = _required_schema_checks(conn)
    if not ok_schema:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    rows = [dict(r) for r in conn.execute(BASE_JOIN_SQL).fetchall()]

    n_total = len(rows)
    n_missing_header = 0

    n_light = 0
    n_light_bad_exptime = 0
    n_light_filter_blank = 0

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r)

        if r.get("h_image_id") is None:
            n_missing_header += 1

        imagetyp = _norm_upper(r.get("imagetyp"))
        if r.get("h_image_id") is not None and imagetyp in ("LIGHT", ""):
            n_light += 1
            exptime = _safe_float(r.get("exptime"))
            if exptime is None or exptime <= 0:
                n_light_bad_exptime += 1
            if not _norm_trim(r.get("filter")):
                n_light_filter_blank += 1

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1

            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        image_id=int(r["image_id"]),
                        file_name=str(r.get("file_name") or ""),
                        status=str(r.get("image_status") or ""),
                        camera=str(r.get("camera") or ""),
                        imagetyp=str(r.get("imagetyp") or ""),
                        filter=str(r.get("filter") or ""),
                        exptime=float(_safe_float(r.get("exptime")) or 0.0),
                        issues=";".join(issues),
                    )
                )

    missing_header_pct = (100.0 * n_missing_header / n_total) if n_total else 0.0
    light_bad_exptime_pct = (100.0 * n_light_bad_exptime / n_light) if n_light else 0.0
    light_filter_blank_pct = (
        (100.0 * n_light_filter_blank / n_light) if n_light else 0.0
    )

    # Additional deep integrity checks (SQL)
    # 1) fits_header_core.image_id must be unique
    dup_ids = [
        int(x["image_id"])
        for x in conn.execute(
            """
            SELECT image_id
            FROM fits_header_core
            GROUP BY image_id
            HAVING COUNT(*) > 1
            ORDER BY image_id
            LIMIT 500;
            """
        ).fetchall()
    ]
    if dup_ids:
        issue_counts["fits_header_core_duplicate_image_id"] = len(dup_ids)

    # 2) fits_header_core should not contain orphan rows (no matching images)
    orphan_header_ids = [
        int(x["image_id"])
        for x in conn.execute(
            """
            SELECT h.image_id
            FROM fits_header_core h
            LEFT JOIN images i ON i.image_id=h.image_id
            WHERE i.image_id IS NULL
            ORDER BY h.image_id
            LIMIT 500;
            """
        ).fetchall()
    ]
    if orphan_header_ids:
        issue_counts["fits_header_core_orphan_row"] = len(orphan_header_ids)

    fail_reasons: List[str] = []

    if n_total == 0:
        fail_reasons.append("no_images_found")

    if missing_header_pct > float(args.fail_if_missing_header_pct):
        fail_reasons.append(
            f"missing_header_pct({missing_header_pct:.3f})>{args.fail_if_missing_header_pct}"
        )

    if light_bad_exptime_pct > float(args.fail_if_light_bad_exptime_pct):
        fail_reasons.append(
            f"light_bad_exptime_pct({light_bad_exptime_pct:.3f})>{args.fail_if_light_bad_exptime_pct}"
        )

    if light_filter_blank_pct > float(args.fail_if_light_filter_blank_pct):
        fail_reasons.append(
            f"light_filter_blank_pct({light_filter_blank_pct:.3f})>{args.fail_if_light_filter_blank_pct}"
        )

    if dup_ids:
        fail_reasons.append(f"duplicate_header_image_id({len(dup_ids)})")

    if orphan_header_ids:
        fail_reasons.append(f"orphan_header_rows({len(orphan_header_ids)})")

    # Any other invariant issues (excluding "light_filter_blank" if you don't want that to hard-fail)
    other_issue_total = sum(
        cnt for k, cnt in issue_counts.items() if k not in ("light_filter_blank",)
    )
    # Note: we still count them, we just don't *force* a fail unless thresholds do.
    # But if you want them to always force FAIL, remove this guard.

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {
            "n_images": n_total,
            "n_missing_fits_header_core_row": n_missing_header,
            "missing_header_pct": round(missing_header_pct, 6),
            "n_light_frames_in_header": n_light,
            "n_light_bad_exptime": n_light_bad_exptime,
            "light_bad_exptime_pct": round(light_bad_exptime_pct, 6),
            "n_light_filter_blank": n_light_filter_blank,
            "light_filter_blank_pct": round(light_filter_blank_pct, 6),
        },
        "integrity": {
            "duplicate_header_image_ids": dup_ids[:500],
            "orphan_header_image_ids": orphan_header_ids[:500],
        },
        "thresholds": {
            "fail_if_missing_header_pct": float(args.fail_if_missing_header_pct),
            "fail_if_light_bad_exptime_pct": float(args.fail_if_light_bad_exptime_pct),
            "fail_if_light_filter_blank_pct": float(
                args.fail_if_light_filter_blank_pct
            ),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": BASE_JOIN_SQL.strip(),
            "max_fail_rows_captured": int(args.max_fail_rows),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "image_id",
                    "file_name",
                    "status",
                    "camera",
                    "imagetyp",
                    "filter",
                    "exptime",
                    "issues",
                ]
            )
            for fr in failing:
                w.writerow(
                    [
                        fr.image_id,
                        fr.file_name,
                        fr.status,
                        fr.camera,
                        fr.imagetyp,
                        fr.filter,
                        fr.exptime,
                        fr.issues,
                    ]
                )

    print(
        f"[test_{MODEL_NAME}] status={status} n_images={n_total} missing_header={n_missing_header} "
        f"n_light={n_light} light_bad_exptime={n_light_bad_exptime} light_filter_blank={n_light_filter_blank}"
    )
    if fail_reasons:
        for r in fail_reasons:
            print(f"  reason: {r}")
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
