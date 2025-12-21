#!/usr/bin/env python3
# centaur/model_tests/models/test_sky_basic_metrics.py
#
# Model test for sky_basic_metrics.
# - Uses a single LEFT JOIN base query (so we cannot "lose" rows in Python)
# - Validates schema + per-image invariants
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Output directory policy:
# - If run individually (no --run-dir):
#     data/model_tests/sky_basic_metrics/test_results_<stamp>/
# - If run under master (master passes --run-dir):
#     <run-dir>/sky_basic_metrics/
#
# Run:
#   python3 centaur/model_tests/models/test_sky_basic_metrics.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_sky_basic_metrics.py --db data/centaurparting.db --run-dir data/model_tests/test_results_20251220_005117

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "sky_basic_metrics"


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
BASE_JOIN_SQL = """
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
  m.db_written_utc,
  m.roi_median_adu,
  m.roi_min_adu,
  m.roi_max_adu,
  m.nan_fraction,
  m.inf_fraction,
  m.clipped_fraction_ff,
  m.clipped_fraction_roi,
  m.roi_fraction,
  m.exptime_s,
  m.roi_median_adu_s,
  m.roi_madstd_adu_s,
  m.ff_median_adu_s,
  m.ff_madstd_adu_s
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN sky_basic_metrics m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
"""


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    # Tables
    for t in ("images", "fits_header_core", "sky_basic_metrics"):
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

    for col in (
        "image_id",
        "expected_fields",
        "read_fields",
        "written_fields",
        "db_written_utc",
        "roi_median_adu",
        "roi_min_adu",
        "roi_max_adu",
        "nan_fraction",
        "inf_fraction",
    ):
        if not _has_col(conn, "sky_basic_metrics", col):
            issues.append(f"sky_basic_metrics_missing_col:{col}")

    return (len(issues) == 0), issues


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # Missing metrics row?
    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        return issues

    ef = _safe_int(r.get("expected_fields"))
    rf = _safe_int(r.get("read_fields"))
    wf = _safe_int(r.get("written_fields"))

    if ef is None or ef <= 0:
        _add_issue(issues, "expected_fields_missing_or_nonpos")
    if rf is None or rf < 0:
        _add_issue(issues, "read_fields_missing_or_negative")
    if wf is None or wf < 0:
        _add_issue(issues, "written_fields_missing_or_negative")
    if ef is not None and rf is not None and rf > ef:
        _add_issue(issues, "read_fields_gt_expected_fields")
    if ef is not None and wf is not None and wf > ef:
        _add_issue(issues, "written_fields_gt_expected_fields")
    if ef is not None and wf is not None and ef > 0 and wf == 0:
        _add_issue(issues, "wrote_zero_fields")

    # nan/inf fractions
    nanf = _safe_float(r.get("nan_fraction"))
    inff = _safe_float(r.get("inf_fraction"))
    if not _clamp01(nanf):
        _add_issue(issues, "nan_fraction_out_of_range")
    if not _clamp01(inff):
        _add_issue(issues, "inf_fraction_out_of_range")

    # clipped fractions (optional)
    cf_ff = _safe_float(r.get("clipped_fraction_ff"))
    cf_roi = _safe_float(r.get("clipped_fraction_roi"))
    if cf_ff is not None and not _clamp01(cf_ff):
        _add_issue(issues, "clipped_fraction_ff_out_of_range")
    if cf_roi is not None and not _clamp01(cf_roi):
        _add_issue(issues, "clipped_fraction_roi_out_of_range")

    # ROI invariants
    vmin = _safe_float(r.get("roi_min_adu"))
    vmax = _safe_float(r.get("roi_max_adu"))
    vmed = _safe_float(r.get("roi_median_adu"))
    if vmin is not None and vmax is not None and vmin > vmax:
        _add_issue(issues, "roi_min_gt_roi_max")
    if vmin is not None and vmax is not None and vmed is not None:
        if not (vmin <= vmed <= vmax):
            _add_issue(issues, "roi_median_outside_minmax")

    # roi_fraction sanity if present
    rfrac = _safe_float(r.get("roi_fraction"))
    if rfrac is not None and not (0.0 < rfrac <= 1.0):
        _add_issue(issues, "roi_fraction_out_of_range")

    # db_written_utc presence
    dbw = r.get("db_written_utc")
    if dbw is None or (isinstance(dbw, str) and not dbw.strip()):
        _add_issue(issues, "db_written_utc_missing")

    # exptime_s consistency if present
    exptime_hdr = _safe_float(r.get("exptime"))
    exptime_s = _safe_float(r.get("exptime_s"))
    if exptime_hdr is not None and exptime_s is not None:
        if abs(exptime_hdr - exptime_s) > 0.01:
            _add_issue(issues, "exptime_s_mismatch_header")

    # scaled seconds should not be negative if present
    for col in (
        "roi_median_adu_s",
        "roi_madstd_adu_s",
        "ff_median_adu_s",
        "ff_madstd_adu_s",
    ):
        v = _safe_float(r.get(col))
        if v is not None and v < 0:
            _add_issue(issues, f"{col}_negative")

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
                "n_wrote_zero_fields": 0,
                "_wf": [],
                "_roi_med": [],
                "_nan": [],
                "_inf": [],
            },
        )
        g["n_frames"] += 1

        if r.get("m_image_id") is None:
            g["n_missing_row"] += 1
            continue

        wf = _safe_int(r.get("written_fields"))
        if wf == 0:
            g["n_wrote_zero_fields"] += 1

        g["_wf"].append(_safe_float(r.get("written_fields")))
        g["_roi_med"].append(_safe_float(r.get("roi_median_adu")))
        g["_nan"].append(_safe_float(r.get("nan_fraction")))
        g["_inf"].append(_safe_float(r.get("inf_fraction")))

    out: List[Dict[str, Any]] = []
    for kk, g in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])
    ):
        out.append(
            {
                "camera": g["camera"],
                "object": g["object"],
                "filter": g["filter"],
                "exptime": g["exptime"],
                "n_frames": g["n_frames"],
                "n_missing_row": g["n_missing_row"],
                "n_wrote_zero_fields": g["n_wrote_zero_fields"],
                "avg_written_fields": (
                    None if avg(g["_wf"]) is None else round(avg(g["_wf"]), 2)
                ),
                "avg_roi_median_adu": (
                    None if avg(g["_roi_med"]) is None else round(avg(g["_roi_med"]), 3)
                ),
                "avg_nan_fraction": (
                    None if avg(g["_nan"]) is None else round(avg(g["_nan"]), 6)
                ),
                "avg_inf_fraction": (
                    None if avg(g["_inf"]) is None else round(avg(g["_inf"]), 6)
                ),
            }
        )
    return out


def _resolve_output_paths(
    *,
    stamp: str,
    out_arg: str,
    csv_arg: str,
    run_dir: str,
) -> Tuple[Path, Path, Path]:
    """
    Returns: (out_dir, out_json, out_csv)

    Rules:
    - If --out/--csv are explicitly provided, respect them exactly.
    - Otherwise choose an output directory based on whether --run-dir is provided:
        * master: <run-dir>/sky_basic_metrics/
        * individual: data/model_tests/sky_basic_metrics/test_results_<stamp>/
      and place default filenames inside it.
    """
    if out_arg.strip():
        out_json = Path(out_arg)
        out_dir = out_json.parent
    else:
        if run_dir.strip():
            out_dir = Path(run_dir) / MODEL_NAME
        else:
            out_dir = (
                Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"
            )
        out_json = out_dir / f"test_{MODEL_NAME}_{stamp}.json"

    if csv_arg.strip():
        out_csv = Path(csv_arg)
        # if user explicitly provided --csv but not --out, keep JSON dir choice; CSV can be elsewhere
    else:
        out_csv = out_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_dir, out_json, out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: sky_basic_metrics")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run output dir (e.g. data/model_tests/test_results_<stamp>). "
        "If provided, this test writes to <run-dir>/sky_basic_metrics/ unless --out/--csv override.",
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

    args = ap.parse_args()

    stamp = _utc_stamp()
    db_path = Path(args.db)

    out_dir, out_json, out_csv = _resolve_output_paths(
        stamp=stamp,
        out_arg=args.out,
        csv_arg=args.csv,
        run_dir=args.run_dir,
    )

    try:
        conn = _connect(db_path)
    except Exception as e:
        # even on ERROR, we still write a JSON somewhere sensible
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "error": f"cannot open db: {e}",
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
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

    # Pull population (join includes metrics)
    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]

    n_total = len(rows)

    # Authoritative missing list via SQL (should match Python counts)
    missing_ids = [
        int(r["image_id"])
        for r in conn.execute(
            """
            SELECT i.image_id
            FROM images i
            JOIN fits_header_core h ON h.image_id=i.image_id
            LEFT JOIN sky_basic_metrics m ON m.image_id=i.image_id
            WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
              AND m.image_id IS NULL
            ORDER BY i.image_id;
            """
        ).fetchall()
    ]

    n_missing = len(missing_ids)

    n_zero_written = 0
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r)

        if r.get("m_image_id") is not None:
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

    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k not in ("missing_metrics_row", "wrote_zero_fields")
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "output_dir": str(out_dir),
        "population": {
            "where": "imagetyp in ('LIGHT','') from fits_header_core",
            "n_total_frames": n_total,
            "n_missing_metrics_row": n_missing,
            "missing_metrics_row_pct": round(missing_pct, 6),
            "missing_image_ids": missing_ids[:500],
            "n_wrote_zero_fields": n_zero_written,
            "wrote_zero_fields_pct": round(zero_written_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
            "fail_if_zero_written_pct": float(args.fail_if_zero_written_pct),
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
        f"[test_{MODEL_NAME}] status={status} n_total={n_total} missing={n_missing} zero_written={n_zero_written}"
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
