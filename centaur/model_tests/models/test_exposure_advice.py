#!/usr/bin/env python3
# centaur/model_tests/models/test_exposure_advice.py
#
# Model test for exposure_advice (table: exposure_advice).
#
# IMPORTANT:
#   This table may NOT contain expected_fields/read_fields/written_fields/db_written_utc.
#   Those fields exist in module_runs for the worker, not necessarily in the metrics table.
#
# What this test does:
# - Single LEFT JOIN base query (prevents losing rows)
# - Worker-run breakdown for exposure_advice_worker:
#     * missing_worker_run
#     * worker_failed
#     * worker_ok_but_missing_metrics   (very important)
# - Deep per-row checks based on actual schema:
#     * required: exposure_advice.image_id
#     * optional: usable/reason conventions
#     * optional: recommended/target exposure fields must be positive if present
#     * dynamic: detect "effectively empty" metrics rows (all metric fields NULL)
#
# Outputs:
#   Individual run (default):
#     data/model_tests/exposure_advice/test_results_<STAMP>/
#   Master run:
#     --run-dir data/model_tests/test_results_<STAMP>
#       writes into: <run-dir>/exposure_advice/
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


TABLE_NAME = "exposure_advice"
WORKER_NAME = "exposure_advice_worker"


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


def _truthy_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


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

    # exposure_advice MUST at least have image_id
    if not _has_col(conn, TABLE_NAME, "image_id"):
        issues.append(f"{TABLE_NAME}_missing_col:image_id")

    return (len(issues) == 0), issues


# -----------------------------
# Dynamic base query builder
# -----------------------------


def _build_base_join_sql(conn: sqlite3.Connection) -> Tuple[str, List[str]]:
    """
    Returns:
      (sql, metric_cols)
    where metric_cols are the real columns in exposure_advice (excluding image_id).
    """
    mcols = _cols(conn, TABLE_NAME)
    metric_cols = [c for c in mcols if c != "image_id"]

    select_metrics: List[str] = ["m.image_id AS m_image_id"]
    for c in metric_cols:
        select_metrics.append(f'm."{c}" AS m_{c}')
    metrics_sql = ",\n  ".join(select_metrics)

    sql = f"""
SELECT
  i.image_id,
  i.file_name,
  i.status AS image_status,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  TRIM(COALESCE(h.object,''))         AS object,
  UPPER(TRIM(COALESCE(h.filter,'')))  AS filter,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp,

  mr.status          AS worker_status,
  mr.expected_fields AS worker_expected_fields,
  mr.read_fields     AS worker_read_fields,
  mr.written_fields  AS worker_written_fields,
  mr.started_utc     AS worker_started_utc,
  mr.ended_utc       AS worker_ended_utc,
  mr.db_written_utc  AS worker_db_written_utc,

  {metrics_sql}
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN module_runs mr
  ON mr.image_id = i.image_id AND mr.module_name = '{WORKER_NAME}'
LEFT JOIN {TABLE_NAME} m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
""".strip()

    return sql, metric_cols


# -----------------------------
# Per-row checks
# -----------------------------


def _m(r: Dict[str, Any], col: str) -> Any:
    return r.get(f"m_{col}")


def _is_metrics_effectively_empty(r: Dict[str, Any], metric_cols: List[str]) -> bool:
    """
    True if all metric columns are NULL/empty.
    We treat a row as "empty" if it has no meaningful values besides image_id.
    """
    if not metric_cols:
        # If table literally only has image_id, we can't judge emptiness.
        return False

    for c in metric_cols:
        v = _m(r, c)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return False
    return True


def _check_one_row(r: Dict[str, Any], metric_cols: List[str]) -> List[str]:
    issues: List[str] = []

    worker_status = r.get("worker_status")
    has_worker_run = worker_status is not None

    if not has_worker_run:
        _add_issue(issues, "missing_worker_run")

    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        if has_worker_run and str(worker_status).upper() == "OK":
            _add_issue(issues, "worker_ok_but_missing_metrics")
        if has_worker_run and str(worker_status).upper() != "OK":
            _add_issue(issues, "worker_failed")
        return issues

    if has_worker_run and str(worker_status).upper() != "OK":
        _add_issue(issues, "worker_failed")

    # Detect "row exists but nothing is populated"
    if _is_metrics_effectively_empty(r, metric_cols):
        _add_issue(issues, "metrics_row_effectively_empty")

    # Deep checks based on optional columns present
    metric_colset = set(metric_cols)

    # usable/reason convention (common pattern in your schema)
    if "usable" in metric_colset:
        u = _safe_int(_m(r, "usable"))
        if u is None:
            _add_issue(issues, "usable_missing")
        elif u not in (0, 1):
            _add_issue(issues, "usable_not_0_or_1")
        if u == 0 and "reason" in metric_colset and not _truthy_str(_m(r, "reason")):
            _add_issue(issues, "reason_missing_when_unusable")

    # Recommended/target exposure fields should be positive if present
    for c in (
        "recommended_exptime_s",
        "recommended_exptime",
        "target_exptime_s",
        "target_exptime",
    ):
        if c in metric_colset:
            v = _safe_float(_m(r, c))
            if v is not None and v <= 0:
                _add_issue(issues, f"{c}_nonpos")

    # Some advice tables store ratios/scores; ensure non-negative if present
    for c in ("score", "eff_score", "time_weight", "confidence"):
        if c in metric_colset:
            v = _safe_float(_m(r, c))
            if v is not None and v < 0:
                _add_issue(issues, f"{c}_negative")

    return issues


# -----------------------------
# Rollup
# -----------------------------


def _rollup(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, float], Dict[str, Any]] = {}

    def k(r: Dict[str, Any]) -> Tuple[str, str, str, float]:
        return (
            str(r.get("camera") or ""),
            str(r.get("object") or ""),
            str(r.get("filter") or ""),
            float(r.get("exptime") or 0.0),
        )

    out: List[Dict[str, Any]] = []
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
                "n_missing_worker_run": 0,
                "n_worker_failed": 0,
                "n_missing_metrics_row": 0,
                "n_worker_ok_but_missing_metrics": 0,
                "n_metrics_row_effectively_empty": 0,
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
                "n_missing_worker_run": g["n_missing_worker_run"],
                "n_worker_failed": g["n_worker_failed"],
                "n_missing_metrics_row": g["n_missing_metrics_row"],
                "n_worker_ok_but_missing_metrics": g["n_worker_ok_but_missing_metrics"],
                "n_metrics_row_effectively_empty": g["n_metrics_row_effectively_empty"],
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

    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run directory (e.g. data/model_tests/test_results_<STAMP>). "
        f"If set, outputs go into <run-dir>/{TABLE_NAME}/",
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

    ap.add_argument(
        "--max-fail-rows",
        type=int,
        default=200,
        help="Max failing image rows to capture",
    )
    ap.add_argument(
        "--fail-if-missing-metrics-pct",
        type=float,
        default=0.0,
        help="Fail if missing metrics row pct > this (0..100)",
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
    ap.add_argument(
        "--fail-if-empty-metrics-row-pct",
        type=float,
        default=0.0,
        help="Fail if metrics_row_effectively_empty pct > this (0..100)",
    )

    args = ap.parse_args()

    stamp = args.stamp.strip() or _utc_stamp()
    db_path = Path(args.db)

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

    base_sql, metric_cols = _build_base_join_sql(conn)

    base_rows = conn.execute(base_sql).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    n_missing_metrics = 0
    n_missing_worker_run = 0
    n_worker_failed = 0
    n_worker_ok_but_missing = 0
    n_empty_metrics_row = 0

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
            if _is_metrics_effectively_empty(r, metric_cols):
                n_empty_metrics_row += 1

        issues = _check_one_row(r, metric_cols)
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

    missing_metrics_pct = pct(n_missing_metrics)
    missing_worker_pct = pct(n_missing_worker_run)
    worker_failed_pct = pct(n_worker_failed)
    ok_but_missing_pct = pct(n_worker_ok_but_missing)
    empty_metrics_pct = pct(n_empty_metrics_row)

    fail_reasons: List[str] = []
    if n_total == 0:
        fail_reasons.append("no_light_frames_found")

    if missing_metrics_pct > float(args.fail_if_missing_metrics_pct):
        fail_reasons.append(
            f"missing_metrics_row_pct({missing_metrics_pct:.3f})>{args.fail_if_missing_metrics_pct}"
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
    if empty_metrics_pct > float(args.fail_if_empty_metrics_row_pct):
        fail_reasons.append(
            f"metrics_row_effectively_empty_pct({empty_metrics_pct:.3f})>{args.fail_if_empty_metrics_row_pct}"
        )

    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k
        not in (
            "missing_metrics_row",
            "missing_worker_run",
            "worker_failed",
            "worker_ok_but_missing_metrics",
            "metrics_row_effectively_empty",
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
            "missing_metrics_row_pct": round(missing_metrics_pct, 6),
            "n_metrics_row_effectively_empty": n_empty_metrics_row,
            "metrics_row_effectively_empty_pct": round(empty_metrics_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_metrics_pct": float(args.fail_if_missing_metrics_pct),
            "fail_if_missing_worker_run_pct": float(
                args.fail_if_missing_worker_run_pct
            ),
            "fail_if_worker_failed_pct": float(args.fail_if_worker_failed_pct),
            "fail_if_worker_ok_but_missing_pct": float(
                args.fail_if_worker_ok_but_missing_pct
            ),
            "fail_if_empty_metrics_row_pct": float(args.fail_if_empty_metrics_row_pct),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "rollup_by_camera_object_filter_exptime": _rollup(rows),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": base_sql,
            "output_dir": str(base_dir),
            "stamp": stamp,
            "max_fail_rows_captured": int(args.max_fail_rows),
            "metric_columns_seen": sorted(metric_cols),
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
        f"empty_metrics_row={n_empty_metrics_row}"
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
