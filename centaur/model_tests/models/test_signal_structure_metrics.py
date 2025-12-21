#!/usr/bin/env python3
# centaur/model_tests/models/test_signal_structure_metrics.py
#
# Model test for signal_structure_metrics.
# - Uses a single LEFT JOIN base query (so we cannot "lose" rows in Python)
# - Validates schema + per-image invariants
# - Distinguishes pipeline/run issues from model/write issues:
#     * missing_worker_run (no module_runs row for signal_structure_worker)
#     * worker_failed
#     * worker_ok_but_metrics_missing (OK run but no metrics row)
#     * missing_metrics_row (generic, but categorized above)
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_signal_structure_metrics.py --db data/centaurparting.db
#
# Master-run layout:
#   python3 ... --run-dir data/model_tests/test_results_YYYYMMDD_HHMMSS
#   -> outputs under <run-dir>/signal_structure_metrics/

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "signal_structure_metrics"
WORKER_NAME = "signal_structure_worker"


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


# Base join:
# - population: LIGHT frames (including imagetyp='')
# - LEFT JOIN metrics
# - LEFT JOIN module_runs for this worker (latest by ended_utc)
BASE_JOIN_SQL = f"""
WITH mr AS (
  SELECT
    image_id,
    module_name,
    status,
    expected_fields,
    read_fields,
    written_fields,
    started_utc,
    ended_utc,
    duration_us,
    db_written_utc,
    ROW_NUMBER() OVER (
      PARTITION BY image_id, module_name
      ORDER BY ended_utc DESC, started_utc DESC
    ) AS rn
  FROM module_runs
  WHERE module_name = '{WORKER_NAME}'
)
SELECT
  i.image_id,
  i.file_name,
  i.status AS image_status,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  TRIM(COALESCE(h.object,''))         AS object,
  UPPER(TRIM(COALESCE(h.filter,'')))  AS filter,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp,

  -- module_runs (nullable if worker never ran)
  mr.status       AS worker_status,
  mr.expected_fields AS worker_expected_fields,
  mr.read_fields     AS worker_read_fields,
  mr.written_fields  AS worker_written_fields,
  mr.started_utc   AS worker_started_utc,
  mr.ended_utc     AS worker_ended_utc,
  mr.duration_us   AS worker_duration_us,
  mr.db_written_utc AS worker_db_written_utc,

  -- metrics (nullable if missing)
  m.image_id AS m_image_id,
  m.expected_fields,
  m.read_fields,
  m.written_fields,
  m.parse_warnings,
  m.db_written_utc,
  m.exptime_s,
  m.ff_p90_minus_p10_adu,
  m.ff_p99_minus_p50_adu,
  m.ff_p999_minus_p50_adu,
  m.ff_p90_minus_p10_adu_s,
  m.ff_p99_minus_p50_adu_s,
  m.ff_p999_minus_p50_adu_s,
  m.ff_madstd_adu_s,
  m.plane_slope_mag_adu_per_tile_s,
  m.saturated_pixel_fraction,
  m.psf_fwhm_px_median,
  m.psf_ecc_median,
  m.psf_usable,
  m.eff_score,
  m.time_weight
FROM fits_header_core h
JOIN images i
  ON i.image_id = h.image_id
LEFT JOIN mr
  ON mr.image_id = h.image_id AND mr.rn = 1
LEFT JOIN signal_structure_metrics m
  ON m.image_id = h.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
"""


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    # Tables
    for t in ("images", "fits_header_core", "module_runs", "signal_structure_metrics"):
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

    for col in ("image_id", "module_name", "status", "started_utc", "ended_utc"):
        if not _has_col(conn, "module_runs", col):
            issues.append(f"module_runs_missing_col:{col}")

    # Metrics table columns (deep coverage)
    required_metrics_cols = [
        "image_id",
        "expected_fields",
        "read_fields",
        "written_fields",
        "db_written_utc",
        "exptime_s",
        "ff_p90_minus_p10_adu",
        "ff_p99_minus_p50_adu",
        "ff_p999_minus_p50_adu",
        "ff_p90_minus_p10_adu_s",
        "ff_p99_minus_p50_adu_s",
        "ff_p999_minus_p50_adu_s",
        "ff_madstd_adu_s",
        "plane_slope_mag_adu_per_tile_s",
        "saturated_pixel_fraction",
        "psf_fwhm_px_median",
        "psf_ecc_median",
        "psf_usable",
        "eff_score",
        "time_weight",
    ]
    for col in required_metrics_cols:
        if not _has_col(conn, "signal_structure_metrics", col):
            issues.append(f"signal_structure_metrics_missing_col:{col}")

    return (len(issues) == 0), issues


def _classify_missing(r: Dict[str, Any], issues: List[str]) -> None:
    """
    If metrics row is missing, classify whether the worker ran and what happened.
    """
    if r.get("m_image_id") is not None:
        return

    worker_status = (r.get("worker_status") or "").strip().upper()

    if not worker_status:
        _add_issue(issues, "missing_worker_run")
        _add_issue(issues, "missing_metrics_row")
        return

    if worker_status == "OK":
        _add_issue(issues, "worker_ok_but_metrics_missing")
        _add_issue(issues, "missing_metrics_row")
        return

    if worker_status == "FAILED":
        _add_issue(issues, "worker_failed")
        _add_issue(issues, "missing_metrics_row")
        return

    _add_issue(issues, f"worker_status_unexpected:{worker_status}")
    _add_issue(issues, "missing_metrics_row")


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # If metrics row missing, classify and stop (cannot run field invariants)
    if r.get("m_image_id") is None:
        _classify_missing(r, issues)
        return issues

    # expected/read/written sanity (table-side)
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

    # db_written_utc presence
    dbw = r.get("db_written_utc")
    if dbw is None or (isinstance(dbw, str) and not dbw.strip()):
        _add_issue(issues, "db_written_utc_missing")

    # exptime_s consistency
    exptime_hdr = _safe_float(r.get("exptime"))
    exptime_s = _safe_float(r.get("exptime_s"))
    if exptime_hdr is not None and exptime_s is not None:
        if abs(exptime_hdr - exptime_s) > 0.01:
            _add_issue(issues, "exptime_s_mismatch_header")

    # Saturated fraction in [0,1]
    satf = _safe_float(r.get("saturated_pixel_fraction"))
    if satf is not None and not _clamp01(satf):
        _add_issue(issues, "saturated_pixel_fraction_out_of_range")

    # Percentile deltas should be >= 0 (if present)
    for col in (
        "ff_p90_minus_p10_adu",
        "ff_p99_minus_p50_adu",
        "ff_p999_minus_p50_adu",
        "ff_p90_minus_p10_adu_s",
        "ff_p99_minus_p50_adu_s",
        "ff_p999_minus_p50_adu_s",
        "ff_madstd_adu_s",
        "plane_slope_mag_adu_per_tile_s",
    ):
        v = _safe_float(r.get(col))
        if v is not None and v < 0:
            _add_issue(issues, f"{col}_negative")

    # PSF fields sanity:
    # - psf_usable should be 0/1 if present
    # - psf_ecc_median typically in [0,1] (allow slight over by noise but flag hard outliers)
    # - psf_fwhm_px_median should be > 0 (if present)
    psf_usable = _safe_int(r.get("psf_usable"))
    if psf_usable is None:
        _add_issue(issues, "psf_usable_missing")
    else:
        if psf_usable not in (0, 1):
            _add_issue(issues, "psf_usable_not_boolean")

    ecc = _safe_float(r.get("psf_ecc_median"))
    if ecc is not None:
        if ecc < 0:
            _add_issue(issues, "psf_ecc_median_negative")
        if ecc > 1.5:
            _add_issue(issues, "psf_ecc_median_unrealistically_high")

    fwhm = _safe_float(r.get("psf_fwhm_px_median"))
    if fwhm is not None and fwhm <= 0:
        _add_issue(issues, "psf_fwhm_px_median_missing_or_nonpos")

    # eff_score and time_weight should be >= 0 if present
    eff = _safe_float(r.get("eff_score"))
    if eff is not None and eff < 0:
        _add_issue(issues, "eff_score_negative")

    tw = _safe_float(r.get("time_weight"))
    if tw is not None and tw < 0:
        _add_issue(issues, "time_weight_negative")

    return issues


def _rollup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group by camera/object/filter/exptime. Roll up completeness and a few key averages.
    """
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
                "n_missing_metrics_row": 0,
                "n_missing_worker_run": 0,
                "n_worker_failed": 0,
                "n_worker_ok_but_metrics_missing": 0,
                "n_psf_usable_null": 0,
                "_eff": [],
                "_tw": [],
                "_satf": [],
            },
        )
        g["n_frames"] += 1

        if r.get("m_image_id") is None:
            g["n_missing_metrics_row"] += 1
            ws = (r.get("worker_status") or "").strip().upper()
            if not ws:
                g["n_missing_worker_run"] += 1
            elif ws == "FAILED":
                g["n_worker_failed"] += 1
            elif ws == "OK":
                g["n_worker_ok_but_metrics_missing"] += 1
            continue

        if r.get("psf_usable") is None:
            g["n_psf_usable_null"] += 1

        g["_eff"].append(_safe_float(r.get("eff_score")))
        g["_tw"].append(_safe_float(r.get("time_weight")))
        g["_satf"].append(_safe_float(r.get("saturated_pixel_fraction")))

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
                "n_missing_metrics_row": g["n_missing_metrics_row"],
                "n_missing_worker_run": g["n_missing_worker_run"],
                "n_worker_failed": g["n_worker_failed"],
                "n_worker_ok_but_metrics_missing": g["n_worker_ok_but_metrics_missing"],
                "n_psf_usable_null": g["n_psf_usable_null"],
                "avg_eff_score": (
                    None if avg(g["_eff"]) is None else round(avg(g["_eff"]), 6)
                ),
                "avg_time_weight": (
                    None if avg(g["_tw"]) is None else round(avg(g["_tw"]), 6)
                ),
                "avg_saturated_pixel_fraction": (
                    None if avg(g["_satf"]) is None else round(avg(g["_satf"]), 6)
                ),
            }
        )
    return out


def _resolve_output_paths(
    args: argparse.Namespace, stamp: str
) -> Tuple[Path, Path, Path]:
    """
    Output layout rules:
      - If --run-dir provided (master run): <run-dir>/<model_name>/
      - Else (individual): data/model_tests/<model_name>/test_results_<stamp>/
    """
    if args.run_dir and str(args.run_dir).strip():
        base_dir = Path(args.run_dir).expanduser().resolve() / MODEL_NAME
    else:
        base_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"

    base_dir.mkdir(parents=True, exist_ok=True)

    out_json = (
        Path(args.out).expanduser().resolve()
        if args.out and str(args.out).strip()
        else base_dir / f"test_{MODEL_NAME}_{stamp}.json"
    )
    out_csv = (
        Path(args.csv).expanduser().resolve()
        if args.csv and str(args.csv).strip()
        else base_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv"
    )

    return base_dir, out_json, out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {MODEL_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # output controls
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run dir (puts outputs under <run-dir>/<model_name>/)",
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

    # thresholds / policy
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
        help="Fail if missing worker_run pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-worker-ok-but-metrics-missing-pct",
        type=float,
        default=0.0,
        help="Fail if worker OK but metrics missing pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-worker-failed-pct",
        type=float,
        default=0.0,
        help="Fail if worker FAILED pct > this (0..100)",
    )
    ap.add_argument(
        "--fail-if-psf_usable_null-pct",
        type=float,
        default=0.0,
        help="Fail if psf_usable NULL pct > this (0..100)",
    )

    args = ap.parse_args()

    stamp = _utc_stamp()
    db_path = Path(args.db)
    base_dir, out_json, out_csv = _resolve_output_paths(args, stamp)

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

    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    # Authoritative SQL missing list for metrics row
    missing_metrics_ids = [
        int(r["image_id"])
        for r in conn.execute(
            """
            SELECT h.image_id
            FROM fits_header_core h
            JOIN images i ON i.image_id=h.image_id
            LEFT JOIN signal_structure_metrics m ON m.image_id=h.image_id
            WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
              AND m.image_id IS NULL
            ORDER BY h.image_id;
            """
        ).fetchall()
    ]

    n_missing_metrics = len(missing_metrics_ids)

    # Counts by failure class
    n_missing_worker_run = 0
    n_worker_failed = 0
    n_worker_ok_but_metrics_missing = 0

    n_zero_written = 0
    n_psf_usable_null = 0

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r)

        # Count worker-run classes if metrics missing
        if r.get("m_image_id") is None:
            ws = (r.get("worker_status") or "").strip().upper()
            if not ws:
                n_missing_worker_run += 1
            elif ws == "FAILED":
                n_worker_failed += 1
            elif ws == "OK":
                n_worker_ok_but_metrics_missing += 1

        # zero-written only if metrics row exists
        if r.get("m_image_id") is not None:
            wf = _safe_int(r.get("written_fields"))
            if wf == 0:
                n_zero_written += 1

            if r.get("psf_usable") is None:
                n_psf_usable_null += 1

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

    def pct(n: int) -> float:
        return (100.0 * float(n) / float(n_total)) if n_total else 0.0

    missing_metrics_pct = pct(n_missing_metrics)
    missing_worker_run_pct = pct(n_missing_worker_run)
    worker_failed_pct = pct(n_worker_failed)
    worker_ok_but_missing_pct = pct(n_worker_ok_but_metrics_missing)
    zero_written_pct = pct(n_zero_written)
    psf_usable_null_pct = pct(n_psf_usable_null)

    # Fail policy
    fail_reasons: List[str] = []

    if n_total == 0:
        fail_reasons.append("no_light_frames_found")

    if missing_metrics_pct > float(args.fail_if_missing_pct):
        fail_reasons.append(
            f"missing_metrics_row_pct({missing_metrics_pct:.3f})>{args.fail_if_missing_pct}"
        )

    if missing_worker_run_pct > float(args.fail_if_missing_worker_run_pct):
        fail_reasons.append(
            f"missing_worker_run_pct({missing_worker_run_pct:.3f})>{args.fail_if_missing_worker_run_pct}"
        )

    if worker_ok_but_missing_pct > float(
        args.fail_if_worker_ok_but_metrics_missing_pct
    ):
        fail_reasons.append(
            f"worker_ok_but_metrics_missing_pct({worker_ok_but_missing_pct:.3f})>{args.fail_if_worker_ok_but_metrics_missing_pct}"
        )

    if worker_failed_pct > float(args.fail_if_worker_failed_pct):
        fail_reasons.append(
            f"worker_failed_pct({worker_failed_pct:.3f})>{args.fail_if_worker_failed_pct}"
        )

    if zero_written_pct > float(args.fail_if_zero_written_pct):
        fail_reasons.append(
            f"wrote_zero_fields_pct({zero_written_pct:.3f})>{args.fail_if_zero_written_pct}"
        )

    if psf_usable_null_pct > float(args.fail_if_psf_usable_null_pct):
        fail_reasons.append(
            f"psf_usable_null_pct({psf_usable_null_pct:.3f})>{args.fail_if_psf_usable_null_pct}"
        )

    # If any other invariant violations exist, fail too (deep sanity)
    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k
        not in (
            "missing_metrics_row",
            "missing_worker_run",
            "worker_failed",
            "worker_ok_but_metrics_missing",
            "wrote_zero_fields",
            "psf_usable_missing",
        )
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "worker": {"module_name": WORKER_NAME},
        "population": {
            "where": "imagetyp in ('LIGHT','') from fits_header_core",
            "n_total_frames": n_total,
        },
        "completeness": {
            "n_missing_metrics_row": n_missing_metrics,
            "missing_metrics_row_pct": round(missing_metrics_pct, 6),
            "missing_metrics_image_ids": missing_metrics_ids[:500],
            "n_missing_worker_run": n_missing_worker_run,
            "missing_worker_run_pct": round(missing_worker_run_pct, 6),
            "n_worker_failed": n_worker_failed,
            "worker_failed_pct": round(worker_failed_pct, 6),
            "n_worker_ok_but_metrics_missing": n_worker_ok_but_metrics_missing,
            "worker_ok_but_metrics_missing_pct": round(worker_ok_but_missing_pct, 6),
            "n_wrote_zero_fields": n_zero_written,
            "wrote_zero_fields_pct": round(zero_written_pct, 6),
            "n_psf_usable_null": n_psf_usable_null,
            "psf_usable_null_pct": round(psf_usable_null_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
            "fail_if_missing_worker_run_pct": float(
                args.fail_if_missing_worker_run_pct
            ),
            "fail_if_worker_ok_but_metrics_missing_pct": float(
                args.fail_if_worker_ok_but_metrics_missing_pct
            ),
            "fail_if_worker_failed_pct": float(args.fail_if_worker_failed_pct),
            "fail_if_zero_written_pct": float(args.fail_if_zero_written_pct),
            "fail_if_psf_usable_null_pct": float(args.fail_if_psf_usable_null_pct),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "rollup_by_camera_object_filter_exptime": _rollup(rows),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": BASE_JOIN_SQL.strip(),
            "output_base_dir": str(base_dir),
            "max_fail_rows_captured": int(args.max_fail_rows),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(str(args.csv).strip())
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

    # Terminal summary
    print(
        f"[test_{MODEL_NAME}] status={status} n_total={n_total} "
        f"missing_metrics={n_missing_metrics} missing_worker_run={n_missing_worker_run} "
        f"worker_failed={n_worker_failed} worker_ok_but_missing={n_worker_ok_but_metrics_missing} "
        f"zero_written={n_zero_written} psf_usable_null={n_psf_usable_null}"
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
