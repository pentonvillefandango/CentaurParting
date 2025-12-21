#!/usr/bin/env python3
# centaur/model_tests/models/test_saturation_metrics.py
#
# Model test for saturation_metrics.
# - Uses a single LEFT JOIN base query (so we cannot "lose" rows in Python)
# - Validates schema + per-image invariants
# - Tracks required-field coverage for saturation_adu explicitly (clean FAIL reason)
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run (individual):
#   python3 centaur/model_tests/models/test_saturation_metrics.py --db data/centaurparting.db
#
# Run (master-style):
#   python3 centaur/model_tests/models/test_saturation_metrics.py --db data/centaurparting.db \
#       --run-dir data/model_tests/test_results_20251220_005117

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "saturation_metrics"
WORKER_NAME = "saturation_worker"


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
        if v != v:  # NaN
            return None
        if v == float("inf") or v == float("-inf"):
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


def _clamp01(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


@dataclass
class FailRow:
    image_id: int
    camera: str
    object: str
    filter: str
    exptime: float
    file_name: str
    issues: str


# --- Base query: LIGHT frames (including imagetyp='') with a LEFT JOIN to metrics
BASE_JOIN_SQL = f"""
SELECT
  i.image_id,
  i.file_name,
  i.status,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  TRIM(COALESCE(h.object,''))         AS object,
  UPPER(TRIM(COALESCE(h.filter,'')))  AS filter,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp,

  -- metrics columns (nullable if missing)
  m.image_id AS m_image_id,
  m.expected_fields,
  m.read_fields,
  m.written_fields,
  m.exptime_s,
  m.parse_warnings,
  m.db_written_utc,

  m.saturation_adu,
  m.max_pixel_adu,
  m.saturated_pixel_count,
  m.saturated_pixel_fraction,
  m.brightest_star_peak_adu,
  m.nan_fraction,
  m.inf_fraction,
  m.usable,
  m.reason
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN {MODEL_NAME} m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
"""


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    # Tables
    for t in ("images", "fits_header_core", MODEL_NAME):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    # Minimal columns we rely on
    for col in ("image_id", "file_name", "status"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    for col in ("image_id", "instrume", "imagetyp", "filter", "exptime", "object"):
        if not _has_col(conn, "fits_header_core", col):
            issues.append(f"fits_header_core_missing_col:{col}")

    # saturation_metrics minimum columns we rely on
    for col in (
        "image_id",
        "db_written_utc",
        "saturation_adu",
        "max_pixel_adu",
        "saturated_pixel_fraction",
        "saturated_pixel_count",
    ):
        if not _has_col(conn, MODEL_NAME, col):
            issues.append(f"{MODEL_NAME}_missing_col:{col}")

    # Optional-but-expected (if present, we validate)
    # (No schema failure if absent; just less deep checking.)
    return (len(issues) == 0), issues


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # Missing metrics row?
    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        return issues

    # Field-count bookkeeping if present (not required in schema check, but validate if present)
    ef = _safe_int(r.get("expected_fields"))
    rf = _safe_int(r.get("read_fields"))
    wf = _safe_int(r.get("written_fields"))
    if ef is not None and ef <= 0:
        _add_issue(issues, "expected_fields_nonpos")
    if rf is not None and rf < 0:
        _add_issue(issues, "read_fields_negative")
    if wf is not None and wf < 0:
        _add_issue(issues, "written_fields_negative")
    if ef is not None and rf is not None and rf > ef:
        _add_issue(issues, "read_fields_gt_expected_fields")
    if ef is not None and wf is not None and wf > ef:
        _add_issue(issues, "written_fields_gt_expected_fields")
    if ef is not None and wf is not None and ef > 0 and wf == 0:
        _add_issue(issues, "wrote_zero_fields")

    # db_written_utc presence (should always exist when row exists)
    dbw = r.get("db_written_utc")
    if dbw is None or (isinstance(dbw, str) and not dbw.strip()):
        _add_issue(issues, "db_written_utc_missing")

    # --- Required metric: saturation_adu ---
    # We *track coverage separately* for headline FAIL reason,
    # but still tag per-row issue for diagnostics.
    sat_adu = _safe_float(r.get("saturation_adu"))
    if sat_adu is None or sat_adu <= 0:
        _add_issue(issues, "saturation_adu_missing_or_nonpos")

    # Other core invariants (deep tests)
    maxpix = _safe_float(r.get("max_pixel_adu"))
    if maxpix is None:
        _add_issue(issues, "max_pixel_adu_missing")
    elif maxpix < 0:
        _add_issue(issues, "max_pixel_adu_negative")

    satcnt = _safe_int(r.get("saturated_pixel_count"))
    if satcnt is None:
        _add_issue(issues, "saturated_pixel_count_missing")
    elif satcnt < 0:
        _add_issue(issues, "saturated_pixel_count_negative")

    satfrac = _safe_float(r.get("saturated_pixel_fraction"))
    if satfrac is None:
        _add_issue(issues, "saturated_pixel_fraction_missing")
    elif not _clamp01(satfrac):
        _add_issue(issues, "saturated_pixel_fraction_out_of_range")

    # brightest_star_peak_adu (if present) should be >=0; optionally <= max_pixel_adu
    bstar = _safe_float(r.get("brightest_star_peak_adu"))
    if bstar is not None and bstar < 0:
        _add_issue(issues, "brightest_star_peak_adu_negative")
    if bstar is not None and maxpix is not None and bstar > maxpix + 1e-6:
        _add_issue(issues, "brightest_star_peak_adu_gt_max_pixel_adu")

    # nan/inf fractions (if present) should be in [0,1]
    nanf = _safe_float(r.get("nan_fraction"))
    inff = _safe_float(r.get("inf_fraction"))
    if nanf is not None and not _clamp01(nanf):
        _add_issue(issues, "nan_fraction_out_of_range")
    if inff is not None and not _clamp01(inff):
        _add_issue(issues, "inf_fraction_out_of_range")

    # usable field rules (if present)
    usable = _safe_int(r.get("usable"))
    if usable is not None and usable not in (0, 1):
        _add_issue(issues, "usable_not_0_or_1")
    if usable == 0:
        reason = r.get("reason")
        if reason is None or (isinstance(reason, str) and not reason.strip()):
            _add_issue(issues, "reason_missing_when_unusable")

    # exptime_s consistency if present
    exptime_hdr = _safe_float(r.get("exptime"))
    exptime_s = _safe_float(r.get("exptime_s"))
    if exptime_hdr is not None and exptime_s is not None:
        if abs(exptime_hdr - exptime_s) > 0.01:
            _add_issue(issues, "exptime_s_mismatch_header")

    return issues


def _rollup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # group by camera/object/filter/exptime
    groups: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}

    def k(r: Dict[str, Any]) -> Tuple[str, str, str, float]:
        return (
            str(r.get("camera") or ""),
            str(r.get("object") or ""),
            str(r.get("filter") or ""),
            float(r.get("exptime") or 0.0),
        )

    def avg(vals: List[Optional[float]]) -> Optional[float]:
        vs = [v for v in vals if v is not None]
        if not vs:
            return None
        return float(sum(vs) / float(len(vs)))

    for r in rows:
        kk = k(r)
        g = groups.setdefault(
            kk,
            {
                "camera": kk[0],
                "object": kk[1],
                "filter": kk[2],
                "exptime": kk[3],
                "n_frames": 0,
                "n_missing_row": 0,
                "n_missing_saturation_adu": 0,
                "_satfrac": [],
                "_maxpix": [],
                "_bstar": [],
                "_nan": [],
                "_inf": [],
                "_usable": [],
            },
        )
        g["n_frames"] += 1

        if r.get("m_image_id") is None:
            g["n_missing_row"] += 1
            continue

        sat_adu = _safe_float(r.get("saturation_adu"))
        if sat_adu is None or sat_adu <= 0:
            g["n_missing_saturation_adu"] += 1

        g["_satfrac"].append(_safe_float(r.get("saturated_pixel_fraction")))
        g["_maxpix"].append(_safe_float(r.get("max_pixel_adu")))
        g["_bstar"].append(_safe_float(r.get("brightest_star_peak_adu")))
        g["_nan"].append(_safe_float(r.get("nan_fraction")))
        g["_inf"].append(_safe_float(r.get("inf_fraction")))

        usable = _safe_int(r.get("usable"))
        if usable in (0, 1):
            g["_usable"].append(float(usable))

    out: List[Dict[str, Any]] = []
    for kk, g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
    ):
        usable_avg = avg(g["_usable"])
        out.append(
            {
                "camera": g["camera"],
                "object": g["object"],
                "filter": g["filter"],
                "exptime": g["exptime"],
                "n_frames": g["n_frames"],
                "n_missing_row": g["n_missing_row"],
                "n_missing_saturation_adu": g["n_missing_saturation_adu"],
                "missing_saturation_adu_pct": round(
                    (
                        (100.0 * g["n_missing_saturation_adu"] / g["n_frames"])
                        if g["n_frames"]
                        else 0.0
                    ),
                    6,
                ),
                "avg_saturated_pixel_fraction": (
                    None if avg(g["_satfrac"]) is None else round(avg(g["_satfrac"]), 8)
                ),
                "avg_max_pixel_adu": (
                    None if avg(g["_maxpix"]) is None else round(avg(g["_maxpix"]), 3)
                ),
                "avg_brightest_star_peak_adu": (
                    None if avg(g["_bstar"]) is None else round(avg(g["_bstar"]), 3)
                ),
                "avg_nan_fraction": (
                    None if avg(g["_nan"]) is None else round(avg(g["_nan"]), 8)
                ),
                "avg_inf_fraction": (
                    None if avg(g["_inf"]) is None else round(avg(g["_inf"]), 8)
                ),
                "usable_rate": (None if usable_avg is None else round(usable_avg, 6)),
            }
        )
    return out


def _resolve_output_dir(args_run_dir: str, stamp: str) -> Path:
    """
    If run individually: data/model_tests/<model>/test_results_<stamp>/
    If run under master: --run-dir data/model_tests/test_results_<stamp>
                         -> outputs go in <run-dir>/<model>/
    """
    if args_run_dir.strip():
        base = Path(args_run_dir).expanduser()
        out_dir = base / MODEL_NAME
    else:
        out_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {MODEL_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run directory (e.g. data/model_tests/test_results_<stamp>). If set, outputs go into <run-dir>/<model>/",
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
        default=200,
        help="Max failing image rows to capture",
    )

    ap.add_argument(
        "--fail-if-missing-pct",
        type=float,
        default=0.0,
        help="Fail if missing metrics row pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-zero-written-pct",
        type=float,
        default=0.0,
        help="Fail if wrote_zero_fields pct > this (0..100)",
    )

    # NEW: required-field coverage threshold for saturation_adu
    ap.add_argument(
        "--fail-if-missing-saturation-adu-pct",
        type=float,
        default=0.0,
        help="Fail if saturation_adu missing/nonpos pct > this (0..100). Default 0 (any missing fails).",
    )

    args = ap.parse_args()

    stamp = _utc_stamp()
    out_dir = _resolve_output_dir(args.run_dir, stamp)

    out_json = (
        Path(args.out)
        if args.out.strip()
        else out_dir / f"test_saturation_metrics_{stamp}.json"
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else out_dir / f"test_saturation_metrics_failures_{stamp}.csv"
    )

    try:
        conn = _connect(Path(args.db))
    except Exception as e:
        print(f"[test_saturation_metrics] ERROR: cannot open db: {e}")
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
        print(f"[test_saturation_metrics] ERROR: schema issues: {schema_issues}")
        print(f"[test_saturation_metrics] wrote_json={out_json}")
        conn.close()
        return 1

    # Pull population (join includes metrics)
    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    # Authoritative missing metrics row list via SQL (should match Python)
    missing_ids = [
        int(r["image_id"])
        for r in conn.execute(
            f"""
            SELECT i.image_id
            FROM images i
            JOIN fits_header_core h ON h.image_id=i.image_id
            LEFT JOIN {MODEL_NAME} m ON m.image_id=i.image_id
            WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
              AND m.image_id IS NULL
            ORDER BY i.image_id;
            """
        ).fetchall()
    ]
    n_missing = len(missing_ids)

    # Count required saturation_adu coverage
    missing_sat_adu_ids: List[int] = []

    n_zero_written = 0
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r)

        # Coverage counters
        if r.get("m_image_id") is not None:
            sat_adu = _safe_float(r.get("saturation_adu"))
            if sat_adu is None or sat_adu <= 0:
                missing_sat_adu_ids.append(int(r["image_id"]))

            wf = _safe_int(r.get("written_fields"))
            if wf == 0:
                n_zero_written += 1

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1

            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        image_id=int(r["image_id"]),
                        camera=str(r.get("camera") or ""),
                        object=str(r.get("object") or ""),
                        filter=str(r.get("filter") or ""),
                        exptime=float(r.get("exptime") or 0.0),
                        file_name=str(r.get("file_name") or ""),
                        issues=";".join(issues),
                    )
                )

    missing_pct = (100.0 * float(n_missing) / float(n_total)) if n_total else 0.0
    zero_written_pct = (
        (100.0 * float(n_zero_written) / float(n_total)) if n_total else 0.0
    )

    n_missing_sat_adu = len(missing_sat_adu_ids)
    missing_sat_adu_pct = (
        (100.0 * float(n_missing_sat_adu) / float(n_total)) if n_total else 0.0
    )

    # --- FAIL policy ---
    fail_reasons: List[str] = []
    if n_total == 0:
        fail_reasons.append("no_light_frames_found")

    if missing_pct > float(args.fail_if_missing_pct):
        fail_reasons.append(
            f"missing_metrics_row_pct({missing_pct:.3f})>{args.fail_if_missing_pct}"
        )

    if zero_written_pct > float(args.fail_if_zero_written_pct):
        fail_reasons.append(
            f"wrote_zero_fields_pct({zero_written_pct:.3f})>{args.fail_if_zero_written_pct}"
        )

    # NEW: clean, explicit reason for saturation_adu coverage
    if missing_sat_adu_pct > float(args.fail_if_missing_saturation_adu_pct):
        fail_reasons.append(
            f"saturation_adu_missing_pct({missing_sat_adu_pct:.3f})>{args.fail_if_missing_saturation_adu_pct}"
        )

    # "Other invariants" bucket: exclude missing-row, wrote-zero-fields, and saturation_adu coverage issue
    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k
        not in (
            "missing_metrics_row",
            "wrote_zero_fields",
            "saturation_adu_missing_or_nonpos",
        )
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(args.db),
        "population": {
            "where": "imagetyp in ('LIGHT','') from fits_header_core",
            "n_total_frames": n_total,
            "n_missing_metrics_row": n_missing,
            "missing_metrics_row_pct": round(missing_pct, 6),
            "missing_image_ids": missing_ids[:500],
            "n_wrote_zero_fields": n_zero_written,
            "wrote_zero_fields_pct": round(zero_written_pct, 6),
            "n_missing_saturation_adu": n_missing_sat_adu,
            "missing_saturation_adu_pct": round(missing_sat_adu_pct, 6),
            "missing_saturation_adu_image_ids": missing_sat_adu_ids[:500],
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
            "fail_if_zero_written_pct": float(args.fail_if_zero_written_pct),
            "fail_if_missing_saturation_adu_pct": float(
                args.fail_if_missing_saturation_adu_pct
            ),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "rollup_by_camera_object_filter_exptime": _rollup(rows),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": BASE_JOIN_SQL.strip(),
            "max_fail_rows_captured": int(args.max_fail_rows),
            "worker_name": WORKER_NAME,
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
                    "camera",
                    "object",
                    "filter",
                    "exptime",
                    "file_name",
                    "issues",
                ]
            )
            for fr in failing:
                w.writerow(
                    [
                        fr.image_id,
                        fr.camera,
                        fr.object,
                        fr.filter,
                        fr.exptime,
                        fr.file_name,
                        fr.issues,
                    ]
                )

    print(
        f"[test_saturation_metrics] status={status} n_total={n_total} missing={n_missing} zero_written={n_zero_written}"
    )
    if fail_reasons:
        for rr in fail_reasons:
            print(f"  reason: {rr}")

    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  issue::{k}={v}")

    print(f"[test_saturation_metrics] wrote_json={out_json}")
    if write_csv:
        print(f"[test_saturation_metrics] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
