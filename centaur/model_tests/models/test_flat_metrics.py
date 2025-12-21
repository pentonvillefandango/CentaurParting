#!/usr/bin/env python3
# centaur/model_tests/models/test_flat_metrics.py
#
# Model test for flat_metrics.
# - Defines population as FITS imagetyp in ('FLAT','')?  (we use robust FLAT matching)
# - Uses a LEFT JOIN base query so we can detect missing metrics rows without losing frames
# - Performs deep per-row invariants on any columns that exist (schema-tolerant)
# - Output directory rules:
#     * If run directly: data/model_tests/flat_metrics/test_results_<stamp>/
#     * If run by master: <run_dir>/flat_metrics/
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_flat_metrics.py --db data/centaurparting.db

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MODEL_NAME = "flat_metrics"
TABLE_NAME = "flat_metrics"


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


def _has_col(cols: List[str], col: str) -> bool:
    return col in set(cols)


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


def _make_out_paths(
    run_dir: Optional[str], stamp: str, out: str, csv_path: str
) -> Tuple[Path, Path, Path]:
    if run_dir and run_dir.strip():
        base_dir = Path(run_dir) / MODEL_NAME
    else:
        base_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"

    base_dir.mkdir(parents=True, exist_ok=True)

    out_json = (
        Path(out) if out.strip() else base_dir / f"test_{MODEL_NAME}_{stamp}.json"
    )
    out_csv = (
        Path(csv_path)
        if csv_path.strip()
        else base_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv"
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return base_dir, out_json, out_csv


# Robust FLAT detection. We treat blank imagetyp as NOT-FLAT for this test.
# If you want blanks included, we can add a flag later.
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

  m.image_id AS m_image_id,
  m.*
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN flat_metrics m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('FLAT','SKYFLAT','DOMEFLAT')
ORDER BY i.image_id;
"""


def _required_schema_checks(
    conn: sqlite3.Connection,
) -> Tuple[bool, List[str], List[str]]:
    issues: List[str] = []
    warnings: List[str] = []

    for t in ("images", "fits_header_core", TABLE_NAME):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues, warnings

    images_cols = _cols(conn, "images")
    fh_cols = _cols(conn, "fits_header_core")
    fm_cols = _cols(conn, TABLE_NAME)

    for col in ("image_id", "file_name", "status"):
        if not _has_col(images_cols, col):
            issues.append(f"images_missing_col:{col}")

    for col in ("image_id", "instrume", "imagetyp", "filter", "exptime", "object"):
        if not _has_col(fh_cols, col):
            issues.append(f"fits_header_core_missing_col:{col}")

    if not _has_col(fm_cols, "image_id"):
        issues.append("flat_metrics_missing_col:image_id")

    # Optional module bookkeeping fields:
    optional_mod_cols = [
        "expected_fields",
        "read_fields",
        "written_fields",
        "db_written_utc",
    ]
    missing_optional = [c for c in optional_mod_cols if not _has_col(fm_cols, c)]
    if missing_optional:
        warnings.append(
            "flat_metrics_missing_optional_module_fields:" + ",".join(missing_optional)
        )

    return (len(issues) == 0), issues, warnings


def _check_one_row(r: Dict[str, Any], fm_cols: List[str]) -> List[str]:
    issues: List[str] = []

    # missing metrics row
    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        return issues

    # If module bookkeeping fields exist, validate them
    if _has_col(fm_cols, "expected_fields"):
        ef = _safe_int(r.get("expected_fields"))
        if ef is None or ef <= 0:
            _add_issue(issues, "expected_fields_missing_or_nonpos")
    if _has_col(fm_cols, "read_fields"):
        rf = _safe_int(r.get("read_fields"))
        if rf is None or rf < 0:
            _add_issue(issues, "read_fields_missing_or_negative")
    if _has_col(fm_cols, "written_fields"):
        wf = _safe_int(r.get("written_fields"))
        if wf is None or wf < 0:
            _add_issue(issues, "written_fields_missing_or_negative")
        if _has_col(fm_cols, "expected_fields"):
            ef = _safe_int(r.get("expected_fields"))
            if ef is not None and wf is not None and wf > ef:
                _add_issue(issues, "written_fields_gt_expected_fields")
        if _has_col(fm_cols, "expected_fields"):
            ef = _safe_int(r.get("expected_fields"))
            if ef is not None and wf is not None and ef > 0 and wf == 0:
                _add_issue(issues, "wrote_zero_fields")

    if _has_col(fm_cols, "db_written_utc"):
        dbw = r.get("db_written_utc")
        if dbw is None or (isinstance(dbw, str) and not dbw.strip()):
            _add_issue(issues, "db_written_utc_missing")

    # Common fraction-like fields: if present, must be in [0,1]
    for col in (
        "nan_fraction",
        "inf_fraction",
        "clipped_fraction",
        "clipped_fraction_ff",
        "clipped_fraction_roi",
        "saturated_pixel_fraction",
        "roi_fraction",
    ):
        if _has_col(fm_cols, col):
            v = _safe_float(r.get(col))
            if v is not None and not _clamp01(v):
                _add_issue(issues, f"{col}_out_of_range")

    # Common min/median/max invariants if present
    for prefix in ("roi", "ff"):
        minc = f"{prefix}_min_adu"
        medc = f"{prefix}_median_adu"
        maxc = f"{prefix}_max_adu"
        if _has_col(fm_cols, minc) and _has_col(fm_cols, maxc):
            vmin = _safe_float(r.get(minc))
            vmax = _safe_float(r.get(maxc))
            if vmin is not None and vmax is not None and vmin > vmax:
                _add_issue(issues, f"{prefix}_min_gt_max")
            if _has_col(fm_cols, medc):
                vmed = _safe_float(r.get(medc))
                if vmin is not None and vmax is not None and vmed is not None:
                    if not (vmin <= vmed <= vmax):
                        _add_issue(issues, f"{prefix}_median_outside_minmax")

    # Exptime consistency if present
    if _has_col(fm_cols, "exptime_s"):
        exptime_hdr = _safe_float(r.get("exptime"))
        exptime_s = _safe_float(r.get("exptime_s"))
        if exptime_hdr is not None and exptime_s is not None:
            if abs(exptime_hdr - exptime_s) > 0.01:
                _add_issue(issues, "exptime_s_mismatch_header")

    # Numeric sanity: any *_adu or *_adu_s fields present should not be NaN/inf (safe_float already filters)
    # But we can flag if they exist and are NULL, which is usually suspicious for a metrics table.
    for c in fm_cols:
        if c.endswith("_adu") or c.endswith("_adu_s"):
            if c == "roi_madstd_adu_s" or c == "ff_madstd_adu_s":
                # these might legitimately be NULL sometimes; keep as soft.
                continue
            if r.get(c) is None:
                _add_issue(issues, f"{c}_missing")

    return issues


def _rollup(rows: List[Dict[str, Any]], fm_cols: List[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}

    def key(r: Dict[str, Any]) -> Tuple[str, str, str, float]:
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
        k = key(r)
        g = groups.setdefault(
            k,
            {
                "camera": k[0],
                "object": k[1],
                "filter": k[2],
                "exptime": k[3],
                "n_frames": 0,
                "n_missing_row": 0,
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

        if _has_col(fm_cols, "written_fields"):
            g["_wf"].append(_safe_float(r.get("written_fields")))
        if _has_col(fm_cols, "roi_median_adu"):
            g["_roi_med"].append(_safe_float(r.get("roi_median_adu")))
        if _has_col(fm_cols, "nan_fraction"):
            g["_nan"].append(_safe_float(r.get("nan_fraction")))
        if _has_col(fm_cols, "inf_fraction"):
            g["_inf"].append(_safe_float(r.get("inf_fraction")))

    out: List[Dict[str, Any]] = []
    for k, g in sorted(
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
                "avg_written_fields": (
                    None
                    if not g["_wf"]
                    else (None if avg(g["_wf"]) is None else round(avg(g["_wf"]), 2))
                ),
                "avg_roi_median_adu": (
                    None
                    if not g["_roi_med"]
                    else (
                        None
                        if avg(g["_roi_med"]) is None
                        else round(avg(g["_roi_med"]), 3)
                    )
                ),
                "avg_nan_fraction": (
                    None
                    if not g["_nan"]
                    else (None if avg(g["_nan"]) is None else round(avg(g["_nan"]), 6))
                ),
                "avg_inf_fraction": (
                    None
                    if not g["_inf"]
                    else (None if avg(g["_inf"]) is None else round(avg(g["_inf"]), 6))
                ),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: flat_metrics")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # master runner support
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="If set, write outputs under <run_dir>/flat_metrics/",
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
        "--max-fail-rows", type=int, default=200, help="Max failing rows to capture"
    )
    ap.add_argument(
        "--fail-if-missing-pct",
        type=float,
        default=0.0,
        help="Fail if missing metrics row pct > this (0..100). Default 0 (any missing fails).",
    )
    ap.add_argument(
        "--fail-if-empty",
        type=int,
        default=1,
        help="If 1, FAIL when no FLAT frames are found. Default 1.",
    )

    args = ap.parse_args()

    stamp = _utc_stamp()
    db_path = Path(args.db)
    _, out_json, out_csv = _make_out_paths(args.run_dir, stamp, args.out, args.csv)

    try:
        conn = _connect(db_path)
    except Exception as e:
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        return 1

    ok_schema, schema_issues, warnings = _required_schema_checks(conn)
    if not ok_schema:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
            "warnings": warnings,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    fm_cols = _cols(conn, TABLE_NAME)

    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    if n_total == 0:
        status = "FAIL" if int(args.fail_if_empty) == 1 else "PASS"
        fail_reasons = ["no_flat_frames_found"] if status == "FAIL" else []
        if status == "PASS":
            warnings.append("no_flat_frames_found")

        result = {
            "test": MODEL_NAME,
            "status": status,
            "db_path": str(db_path),
            "population": {
                "where": "imagetyp in ('FLAT','SKYFLAT','DOMEFLAT')",
                "n_total_frames": n_total,
            },
            "fail_reasons": fail_reasons,
            "warnings": warnings,
            "issue_counts": {},
            "rollup_by_camera_object_filter_exptime": [],
            "failing_examples": [],
            "notes": {
                "base_query": BASE_JOIN_SQL.strip(),
                "max_fail_rows_captured": int(args.max_fail_rows),
            },
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] status={status} n_total={n_total} missing=0")
        for w in warnings:
            print(f"  warn: {w}")
        if fail_reasons:
            for r in fail_reasons:
                print(f"  reason: {r}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 0 if status == "PASS" else 2

    n_missing = 0
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r, fm_cols)

        if r.get("m_image_id") is None:
            n_missing += 1

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

    fail_reasons: List[str] = []
    if missing_pct > float(args.fail_if_missing_pct):
        fail_reasons.append(
            f"missing_metrics_row_pct({missing_pct:.3f})>{args.fail_if_missing_pct}"
        )

    other_issue_total = sum(
        cnt for k, cnt in issue_counts.items() if k != "missing_metrics_row"
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {
            "where": "imagetyp in ('FLAT','SKYFLAT','DOMEFLAT') from fits_header_core",
            "n_total_frames": n_total,
            "n_missing_metrics_row": n_missing,
            "missing_metrics_row_pct": round(missing_pct, 6),
        },
        "thresholds": {"fail_if_missing_pct": float(args.fail_if_missing_pct)},
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "rollup_by_camera_object_filter_exptime": _rollup(rows, fm_cols),
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

    print(f"[test_{MODEL_NAME}] status={status} n_total={n_total} missing={n_missing}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"  reason: {r}")
    if warnings:
        for w in warnings[:10]:
            print(f"  warn: {w}")
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
