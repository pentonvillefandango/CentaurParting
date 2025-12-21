#!/usr/bin/env python3
# centaur/model_tests/models/test_roi_signal_metrics.py
#
# Model test for roi_signal_metrics.
# - Uses a single LEFT JOIN base query (so we cannot "lose" rows in Python)
# - Validates schema + per-image invariants
# - Explicitly reports worker-run breakdown for roi_signal_worker:
#     * missing_worker_run
#     * worker_failed
#     * worker_ok_but_missing_metrics   (important)
#
# Outputs (always JSON, CSV on FAIL or if --csv provided):
#   Individual run (default):
#     data/model_tests/roi_signal_metrics/test_results_<STAMP>/
#   Master run:
#     --run-dir data/model_tests/test_results_<STAMP>
#       writes into: <run-dir>/roi_signal_metrics/
#
# Exit codes:
#   0 PASS
#   2 FAIL
#   1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_roi_signal_metrics.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_roi_signal_metrics.py --db data/centaurparting.db --run-dir data/model_tests/test_results_20251220_005117

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TABLE_NAME = "roi_signal_metrics"
WORKER_NAME = "roi_signal_worker"


# -----------------------------
# Helpers
# -----------------------------


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


def _truthy_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


# -----------------------------
# Data model
# -----------------------------


@dataclass
class FailRow:
    image_id: int
    camera: str
    object: str
    filter: str
    exptime: float
    file_name: str
    issues: str


# -----------------------------
# Base query (population + metrics + worker-run)
# -----------------------------

BASE_JOIN_SQL = f"""
SELECT
  i.image_id,
  i.file_name,
  i.status AS image_status,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  TRIM(COALESCE(h.object,''))         AS object,
  UPPER(TRIM(COALESCE(h.filter,'')))  AS filter,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp,

  -- worker-run row (nullable if missing)
  mr.status        AS worker_status,
  mr.expected_fields AS worker_expected_fields,
  mr.read_fields     AS worker_read_fields,
  mr.written_fields  AS worker_written_fields,
  mr.started_utc   AS worker_started_utc,
  mr.ended_utc     AS worker_ended_utc,
  mr.db_written_utc AS worker_db_written_utc,

  -- metrics columns (nullable if missing)
  m.image_id AS m_image_id,
  m.expected_fields,
  m.read_fields,
  m.written_fields,
  m.db_written_utc,

  m.exptime_s,
  m.roi_kind,
  m.roi_fraction,
  m.roi_bbox_json,
  m.bg_bbox_json,

  m.obj_median_adu,
  m.obj_madstd_adu,
  m.bg_median_adu,
  m.bg_madstd_adu,
  m.obj_minus_bg_adu,
  m.obj_minus_bg_adu_s,

  m.clipped_fraction_obj,
  m.clipped_fraction_bg,

  m.parse_warnings,
  m.usable,
  m.reason
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN module_runs mr
  ON mr.image_id = i.image_id AND mr.module_name = '{WORKER_NAME}'
LEFT JOIN {TABLE_NAME} m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
"""


# -----------------------------
# Schema checks
# -----------------------------


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    for t in ("images", "fits_header_core", "module_runs", TABLE_NAME):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    for col in ("image_id", "file_name", "status"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    for col in ("image_id", "instrume", "imagetyp", "filter", "exptime", "object"):
        if not _has_col(conn, "fits_header_core", col):
            issues.append(f"fits_header_core_missing_col:{col}")

    for col in (
        "image_id",
        "module_name",
        "status",
        "started_utc",
        "ended_utc",
        "db_written_utc",
    ):
        if not _has_col(conn, "module_runs", col):
            issues.append(f"module_runs_missing_col:{col}")

    # Core roi_signal_metrics columns we rely on for deep checks
    for col in (
        "image_id",
        "expected_fields",
        "read_fields",
        "written_fields",
        "db_written_utc",
        "exptime_s",
        "roi_kind",
        "roi_fraction",
        "obj_median_adu",
        "bg_median_adu",
        "obj_minus_bg_adu",
        "usable",
        "reason",
    ):
        if not _has_col(conn, TABLE_NAME, col):
            issues.append(f"{TABLE_NAME}_missing_col:{col}")

    return (len(issues) == 0), issues


# -----------------------------
# Per-row checks
# -----------------------------


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # Worker-run breakdown signals (these become headline counters too)
    worker_status = r.get("worker_status")
    has_worker_run = worker_status is not None

    if not has_worker_run:
        _add_issue(issues, "missing_worker_run")

    # Missing metrics row?
    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        # If worker ran OK but metrics missing, call that out explicitly (important bucket)
        if has_worker_run and str(worker_status).upper() == "OK":
            _add_issue(issues, "worker_ok_but_missing_metrics")
        # If worker ran but not OK, also categorize
        if has_worker_run and str(worker_status).upper() != "OK":
            _add_issue(issues, "worker_failed")
        return issues

    # If worker ran and not OK, it’s still a problem even if metrics exist (usually shouldn’t happen)
    if has_worker_run and str(worker_status).upper() != "OK":
        _add_issue(issues, "worker_failed")

    # fields sanity
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

    # db_written_utc
    if not _truthy_str(r.get("db_written_utc")):
        _add_issue(issues, "db_written_utc_missing")

    # exptime consistency
    exptime_hdr = _safe_float(r.get("exptime"))
    exptime_s = _safe_float(r.get("exptime_s"))
    if exptime_hdr is not None and exptime_s is not None:
        if abs(exptime_hdr - exptime_s) > 0.01:
            _add_issue(issues, "exptime_s_mismatch_header")

    # roi_kind
    roi_kind = r.get("roi_kind")
    if roi_kind is None or (isinstance(roi_kind, str) and not roi_kind.strip()):
        _add_issue(issues, "roi_kind_missing_or_empty")

    # roi_fraction in (0,1]
    rfrac = _safe_float(r.get("roi_fraction"))
    if rfrac is None:
        _add_issue(issues, "roi_fraction_missing")
    else:
        if not (0.0 < rfrac <= 1.0):
            _add_issue(issues, "roi_fraction_out_of_range")

    # clipped fractions in [0,1] if present
    cf_obj = _safe_float(r.get("clipped_fraction_obj"))
    cf_bg = _safe_float(r.get("clipped_fraction_bg"))
    if cf_obj is not None and not _clamp01(cf_obj):
        _add_issue(issues, "clipped_fraction_obj_out_of_range")
    if cf_bg is not None and not _clamp01(cf_bg):
        _add_issue(issues, "clipped_fraction_bg_out_of_range")

    # Stats sanity: medians can be any real, MADSTD should be >= 0 if present
    obj_mad = _safe_float(r.get("obj_madstd_adu"))
    bg_mad = _safe_float(r.get("bg_madstd_adu"))
    if obj_mad is not None and obj_mad < 0:
        _add_issue(issues, "obj_madstd_negative")
    if bg_mad is not None and bg_mad < 0:
        _add_issue(issues, "bg_madstd_negative")

    # obj_minus_bg should match median difference approximately (when all are present)
    obj_med = _safe_float(r.get("obj_median_adu"))
    bg_med = _safe_float(r.get("bg_median_adu"))
    omb = _safe_float(r.get("obj_minus_bg_adu"))
    if obj_med is None:
        _add_issue(issues, "obj_median_missing")
    if bg_med is None:
        _add_issue(issues, "bg_median_missing")
    if omb is None:
        _add_issue(issues, "obj_minus_bg_missing")
    if obj_med is not None and bg_med is not None and omb is not None:
        # Robust pipelines sometimes round; allow small tolerance
        if abs((obj_med - bg_med) - omb) > 0.5:
            _add_issue(issues, "obj_minus_bg_not_close_to_objmed_minus_bgmed")

    # obj_minus_bg_adu_s should be approx omb / exptime_s (if all present)
    omb_s = _safe_float(r.get("obj_minus_bg_adu_s"))
    if (
        omb_s is not None
        and exptime_s is not None
        and exptime_s > 0
        and omb is not None
    ):
        if abs((omb / exptime_s) - omb_s) > 0.01:
            _add_issue(
                issues, "obj_minus_bg_adu_s_not_close_to_obj_minus_bg_adu_over_exptime"
            )

    # usable / reason relationship
    usable = _safe_int(r.get("usable"))
    reason = r.get("reason")
    if usable is None:
        _add_issue(issues, "usable_missing")
    else:
        if usable not in (0, 1):
            _add_issue(issues, "usable_not_0_or_1")
        if usable == 0 and not _truthy_str(reason):
            _add_issue(issues, "reason_missing_when_unusable")

    return issues


# -----------------------------
# Rollups
# -----------------------------


def _rollup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by camera/object/filter/exptime/roi_kind.
    """
    groups: Dict[Tuple[str, str, str, float, str], Dict[str, Any]] = {}

    def k(r: Dict[str, Any]) -> Tuple[str, str, str, float, str]:
        return (
            str(r.get("camera") or ""),
            str(r.get("object") or ""),
            str(r.get("filter") or ""),
            float(r.get("exptime") or 0.0),
            str(r.get("roi_kind") or ""),
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
                "roi_kind": kk[4],
                "n_frames": 0,
                "n_missing_worker_run": 0,
                "n_worker_failed": 0,
                "n_missing_metrics_row": 0,
                "n_worker_ok_but_missing_metrics": 0,
                "n_wrote_zero_fields": 0,
                "_wf": [],
                "_roi_frac": [],
                "_omb": [],
            },
        )
        g["n_frames"] += 1

        worker_status = r.get("worker_status")
        if worker_status is None:
            g["n_missing_worker_run"] += 1
        else:
            if str(worker_status).upper() != "OK":
                g["n_worker_failed"] += 1

        if r.get("m_image_id") is None:
            g["n_missing_metrics_row"] += 1
            if worker_status is not None and str(worker_status).upper() == "OK":
                g["n_worker_ok_but_missing_metrics"] += 1
            continue

        wf = _safe_int(r.get("written_fields"))
        if wf == 0:
            g["n_wrote_zero_fields"] += 1

        g["_wf"].append(_safe_float(r.get("written_fields")))
        g["_roi_frac"].append(_safe_float(r.get("roi_fraction")))
        g["_omb"].append(_safe_float(r.get("obj_minus_bg_adu")))

    out: List[Dict[str, Any]] = []
    for kk, g in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3], kv[0][4]),
    ):
        out.append(
            {
                "camera": g["camera"],
                "object": g["object"],
                "filter": g["filter"],
                "exptime": g["exptime"],
                "roi_kind": g["roi_kind"],
                "n_frames": g["n_frames"],
                "n_missing_worker_run": g["n_missing_worker_run"],
                "n_worker_failed": g["n_worker_failed"],
                "n_missing_metrics_row": g["n_missing_metrics_row"],
                "n_worker_ok_but_missing_metrics": g["n_worker_ok_but_missing_metrics"],
                "n_wrote_zero_fields": g["n_wrote_zero_fields"],
                "avg_written_fields": (
                    None if avg(g["_wf"]) is None else round(avg(g["_wf"]), 2)
                ),
                "avg_roi_fraction": (
                    None
                    if avg(g["_roi_frac"]) is None
                    else round(avg(g["_roi_frac"]), 6)
                ),
                "avg_obj_minus_bg_adu": (
                    None if avg(g["_omb"]) is None else round(avg(g["_omb"]), 3)
                ),
            }
        )
    return out


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {TABLE_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # Output routing
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run directory (e.g. data/model_tests/test_results_<STAMP>). "
        "If set, outputs go into <run-dir>/roi_signal_metrics/",
    )
    ap.add_argument(
        "--stamp",
        type=str,
        default="",
        help="Optional stamp override (YYYYMMDD_HHMMSS). If blank, generated now.",
    )
    ap.add_argument(
        "--out", type=str, default="", help="Output JSON path (optional override)"
    )
    ap.add_argument(
        "--csv", type=str, default="", help="Failures CSV path (optional override)"
    )

    # Policy thresholds
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
    ap.add_argument(
        "--fail-if-missing-worker-run-pct",
        type=float,
        default=0.0,
        help="Fail if missing worker run pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-worker-failed-pct",
        type=float,
        default=0.0,
        help="Fail if worker failed pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-worker-ok-but-missing-pct",
        type=float,
        default=0.0,
        help="Fail if worker OK but missing metrics pct > this (0..100)",
    )

    args = ap.parse_args()

    stamp = args.stamp.strip() or _utc_stamp()
    db_path = Path(args.db)

    # Output directory selection
    if args.run_dir.strip():
        base_dir = Path(args.run_dir) / TABLE_NAME
    else:
        base_dir = Path("data") / "model_tests" / TABLE_NAME / f"test_results_{stamp}"

    base_dir.mkdir(parents=True, exist_ok=True)

    out_json = (
        Path(args.out)
        if args.out.strip()
        else (base_dir / f"test_{TABLE_NAME}_{stamp}.json")
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else (base_dir / f"test_{TABLE_NAME}_failures_{stamp}.csv")
    )

    try:
        conn = _connect(db_path)
    except Exception as e:
        print(f"[test_{TABLE_NAME}] ERROR: cannot open db: {e}")
        return 1

    ok_schema, schema_issues = _required_schema_checks(conn)
    if not ok_schema:
        result = {
            "test": TABLE_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{TABLE_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{TABLE_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    # Headline counters
    n_missing_metrics = 0
    n_zero_written = 0
    n_missing_worker_run = 0
    n_worker_failed = 0
    n_worker_ok_but_missing = 0

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        worker_status = r.get("worker_status")
        if worker_status is None:
            n_missing_worker_run += 1
        else:
            if str(worker_status).upper() != "OK":
                n_worker_failed += 1

        if r.get("m_image_id") is None:
            n_missing_metrics += 1
            if worker_status is not None and str(worker_status).upper() == "OK":
                n_worker_ok_but_missing += 1
        else:
            wf = _safe_int(r.get("written_fields"))
            if wf == 0:
                n_zero_written += 1

        issues = _check_one_row(r)
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

    def pct(x: int) -> float:
        return (100.0 * float(x) / float(n_total)) if n_total else 0.0

    missing_pct = pct(n_missing_metrics)
    zero_written_pct = pct(n_zero_written)
    missing_worker_pct = pct(n_missing_worker_run)
    worker_failed_pct = pct(n_worker_failed)
    ok_but_missing_pct = pct(n_worker_ok_but_missing)

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
    if missing_worker_pct > float(args.fail_if_missing_worker_run_pct):
        fail_reasons.append(
            f"missing_worker_run_pct({missing_worker_pct:.3f})>{args.fail_if_missing_worker_run_pct}"
        )
    if worker_failed_pct > float(args.fail_if_worker_failed_pct):
        fail_reasons.append(
            f"worker_failed_pct({worker_failed_pct:.3f})>{args.fail_if_worker_failed_pct}"
        )
    if ok_but_missing_pct > float(args.fail_if_worker_ok_but_missing_pct):
        fail_reasons.append(
            f"worker_ok_but_missing_metrics_pct({ok_but_missing_pct:.3f})>{args.fail_if_worker_ok_but_missing_pct}"
        )

    # Any other invariant violations beyond the "bucket" flags should fail too
    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k
        not in (
            "missing_metrics_row",
            "missing_worker_run",
            "worker_failed",
            "worker_ok_but_missing_metrics",
            "wrote_zero_fields",
        )
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": TABLE_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {
            "where": "fits_header_core.imagetyp in ('LIGHT','')",
            "n_total_frames": n_total,
        },
        "worker": {
            "module_name": WORKER_NAME,
            "n_missing_worker_run": n_missing_worker_run,
            "missing_worker_run_pct": round(missing_worker_pct, 6),
            "n_worker_failed": n_worker_failed,
            "worker_failed_pct": round(worker_failed_pct, 6),
            "n_worker_ok_but_missing_metrics": n_worker_ok_but_missing,
            "worker_ok_but_missing_metrics_pct": round(ok_but_missing_pct, 6),
        },
        "metrics": {
            "n_missing_metrics_row": n_missing_metrics,
            "missing_metrics_row_pct": round(missing_pct, 6),
            "n_wrote_zero_fields": n_zero_written,
            "wrote_zero_fields_pct": round(zero_written_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
            "fail_if_zero_written_pct": float(args.fail_if_zero_written_pct),
            "fail_if_missing_worker_run_pct": float(
                args.fail_if_missing_worker_run_pct
            ),
            "fail_if_worker_failed_pct": float(args.fail_if_worker_failed_pct),
            "fail_if_worker_ok_but_missing_pct": float(
                args.fail_if_worker_ok_but_missing_pct
            ),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "rollup_by_camera_object_filter_exptime_roi_kind": _rollup(rows),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": BASE_JOIN_SQL.strip(),
            "output_dir": str(base_dir),
            "stamp": stamp,
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
        f"[test_{TABLE_NAME}] status={status} n_total={n_total} "
        f"missing_metrics={n_missing_metrics} missing_worker_run={n_missing_worker_run} "
        f"worker_failed={n_worker_failed} worker_ok_but_missing={n_worker_ok_but_missing} "
        f"zero_written={n_zero_written}"
    )
    if fail_reasons:
        for rr in fail_reasons:
            print(f"  reason: {rr}")
    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  issue::{k}={v}")

    print(f"[test_{TABLE_NAME}] wrote_json={out_json}")
    if write_csv:
        print(f"[test_{TABLE_NAME}] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
