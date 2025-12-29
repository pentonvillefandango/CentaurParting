#!/usr/bin/env python3
# centaur/model_tests/models/test_module_runs.py
#
# System test for module_runs coverage on LIGHT frames.
# Reports, per worker:
#   - missing_worker_run
#   - worker_failed (status != OK on latest run)
#   - worker_ok_but_missing_metrics (worker OK but metrics row missing)
#
# Outputs:
#   * JSON results (always)
#   * CSV failures (on FAIL, or if --csv provided)
#
# Exit codes:
#   0 PASS
#   2 FAIL
#   1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_module_runs.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_module_runs.py --db data/centaurparting.db --run-dir data/model_tests/test_results_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "module_runs"


EXPECTED_WORKERS: List[str] = [
    "fits_header_worker",
    "sky_basic_worker",
    "sky_background2d_worker",
    "nebula_mask_worker",
    "saturation_worker",
    "signal_structure_worker",
    "roi_signal_worker",
    "psf_detect_worker",
    "psf_basic_worker",
    "star_headroom_worker",
    "psf_grid_worker",
    "psf_model_worker",
    "masked_signal_worker",
    "exposure_advice_worker",
    "frame_quality_worker",
]

# Worker -> metrics table that should have image_id if worker ran OK
WORKER_TO_METRICS_TABLE: Dict[str, str] = {
    "fits_header_worker": "fits_header_core",
    "sky_basic_worker": "sky_basic_metrics",
    "sky_background2d_worker": "sky_background2d_metrics",
    "nebula_mask_worker": "nebula_mask_metrics",
    "saturation_worker": "saturation_metrics",
    "signal_structure_worker": "signal_structure_metrics",
    "roi_signal_worker": "roi_signal_metrics",
    "psf_detect_worker": "psf_detect_metrics",
    "psf_basic_worker": "psf_basic_metrics",
    "star_headroom_worker": "star_headroom_metrics",
    "psf_grid_worker": "psf_grid_metrics",
    "psf_model_worker": "psf_model_metrics",
    "masked_signal_worker": "masked_signal_metrics",
    "exposure_advice_worker": "exposure_advice",
    "frame_quality_worker": "frame_quality_metrics",
}


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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


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


def _norm_upper(x: Any) -> str:
    return str(x or "").strip().upper()


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
    camera: str
    object: str
    filter: str
    exptime: float
    worker: str
    issues: str


# Population: LIGHT frames
LIGHT_BASE_SQL = """
SELECT
  h.image_id,
  i.file_name,
  UPPER(TRIM(COALESCE(h.instrume,''))) AS camera,
  TRIM(COALESCE(h.object,''))         AS object,
  UPPER(TRIM(COALESCE(h.filter,'')))  AS filter,
  CAST(COALESCE(h.exptime, 0) AS REAL) AS exptime,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp
FROM fits_header_core h
JOIN images i ON i.image_id=h.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY h.image_id;
"""


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    for t in ("images", "fits_header_core", "module_runs"):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    # needed for population
    for col in ("image_id", "file_name"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    for col in ("image_id", "imagetyp", "instrume", "filter", "exptime", "object"):
        if not _has_col(conn, "fits_header_core", col):
            issues.append(f"fits_header_core_missing_col:{col}")

    # needed for module_runs evaluation
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

    # metrics tables referenced by mapping (only check existence, not full schema)
    for worker, tbl in WORKER_TO_METRICS_TABLE.items():
        if not _table_exists(conn, tbl):
            issues.append(f"missing_table_for_worker:{worker}:{tbl}")

    return (len(issues) == 0), issues


def _load_latest_runs_for_light_images(
    conn: sqlite3.Connection, light_ids: List[int]
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    Returns dict keyed by (image_id, module_name) -> row dict of the "latest" run.
    Latest chosen by:
      - ended_utc DESC if present else started_utc DESC else rowid DESC
    """
    latest: Dict[Tuple[int, str], Dict[str, Any]] = {}
    if not light_ids:
        return latest

    # chunk to avoid SQLite parameter limits
    CHUNK = 900
    for i0 in range(0, len(light_ids), CHUNK):
        chunk = light_ids[i0 : i0 + CHUNK]
        qmarks = ",".join(["?"] * len(chunk))
        rows = conn.execute(
            f"""
            SELECT rowid, *
            FROM module_runs
            WHERE image_id IN ({qmarks})
              AND module_name IN ({",".join(["?"] * len(EXPECTED_WORKERS))})
            """,
            tuple(chunk) + tuple(EXPECTED_WORKERS),
        ).fetchall()

        for rr in rows:
            r = dict(rr)
            key = (int(r["image_id"]), str(r["module_name"]))
            # compare to choose latest
            prev = latest.get(key)
            if prev is None:
                latest[key] = r
                continue

            def key_tuple(x: Dict[str, Any]) -> Tuple[int, str, str, int]:
                # Use string ordering of ISO timestamps (works)
                ended = str(x.get("ended_utc") or "")
                started = str(x.get("started_utc") or "")
                rowid = int(_safe_int(x.get("rowid")) or 0)
                # Prefer ended_utc, then started_utc, then rowid
                return (1 if ended else 0, ended or started, started, rowid)

            if key_tuple(r) > key_tuple(prev):
                latest[key] = r

    return latest


def _metrics_row_exists(conn: sqlite3.Connection, table: str, image_id: int) -> bool:
    r = conn.execute(
        f"SELECT 1 FROM {table} WHERE image_id=? LIMIT 1;",
        (image_id,),
    ).fetchone()
    return bool(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="System test: module_runs")
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
        default=600,
        help="Max failing rows to include in JSON/CSV",
    )

    ap.add_argument(
        "--fail-if-missing-worker-run-pct",
        type=float,
        default=0.0,
        help="Fail if missing_worker_run pct > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-worker-failed-pct",
        type=float,
        default=0.0,
        help="Fail if worker_failed pct > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-worker-ok-but-metrics-missing-pct",
        type=float,
        default=0.0,
        help="Fail if worker_ok_but_missing_metrics pct > this (0..100). Default 0.",
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

    light_rows = [dict(r) for r in conn.execute(LIGHT_BASE_SQL).fetchall()]
    light_ids = [int(r["image_id"]) for r in light_rows]
    n_total = len(light_ids)

    latest_runs = _load_latest_runs_for_light_images(conn, light_ids)

    # counters (overall, aggregated over image_id x worker)
    n_pairs = n_total * len(EXPECTED_WORKERS) if n_total else 0
    n_missing_run = 0
    n_failed = 0
    n_ok_but_missing_metrics = 0

    # rollups per worker
    per_worker: Dict[str, Dict[str, Any]] = {
        w: {
            "worker": w,
            "n_light": n_total,
            "missing_worker_run": 0,
            "worker_failed": 0,
            "worker_ok_but_missing_metrics": 0,
        }
        for w in EXPECTED_WORKERS
    }

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    # quick lookup for row metadata
    base_by_id: Dict[int, Dict[str, Any]] = {int(r["image_id"]): r for r in light_rows}

    for image_id in light_ids:
        base = base_by_id[image_id]

        for worker in EXPECTED_WORKERS:
            issues: List[str] = []
            run = latest_runs.get((image_id, worker))

            if run is None:
                n_missing_run += 1
                per_worker[worker]["missing_worker_run"] += 1
                _add_issue(issues, "missing_worker_run")
            else:
                status = _norm_upper(run.get("status"))
                if status != "OK":
                    n_failed += 1
                    per_worker[worker]["worker_failed"] += 1
                    _add_issue(issues, f"worker_failed(status={status})")

                # basic run record sanity
                if not str(run.get("started_utc") or "").strip():
                    _add_issue(issues, "started_utc_missing")
                if not str(run.get("ended_utc") or "").strip():
                    _add_issue(issues, "ended_utc_missing")
                if not str(run.get("db_written_utc") or "").strip():
                    _add_issue(issues, "db_written_utc_missing")

                # If OK, require metrics row exists
                if status == "OK":
                    metrics_table = WORKER_TO_METRICS_TABLE.get(worker, "")
                    if metrics_table:
                        ok = _metrics_row_exists(conn, metrics_table, image_id)
                        if not ok:
                            n_ok_but_missing_metrics += 1
                            per_worker[worker]["worker_ok_but_missing_metrics"] += 1
                            _add_issue(
                                issues,
                                f"worker_ok_but_missing_metrics(table={metrics_table})",
                            )

            if issues:
                for it in issues:
                    issue_counts[it] = issue_counts.get(it, 0) + 1

                if len(failing) < int(args.max_fail_rows):
                    failing.append(
                        FailRow(
                            image_id=image_id,
                            file_name=str(base.get("file_name") or ""),
                            camera=str(base.get("camera") or ""),
                            object=str(base.get("object") or ""),
                            filter=str(base.get("filter") or ""),
                            exptime=float(_safe_float(base.get("exptime")) or 0.0),
                            worker=worker,
                            issues=";".join(issues),
                        )
                    )

    missing_run_pct = (100.0 * n_missing_run / n_pairs) if n_pairs else 0.0
    failed_pct = (100.0 * n_failed / n_pairs) if n_pairs else 0.0
    ok_missing_metrics_pct = (
        (100.0 * n_ok_but_missing_metrics / n_pairs) if n_pairs else 0.0
    )

    fail_reasons: List[str] = []
    if n_total == 0:
        fail_reasons.append("no_light_frames_found")

    if missing_run_pct > float(args.fail_if_missing_worker_run_pct):
        fail_reasons.append(
            f"missing_worker_run_pct({missing_run_pct:.3f})>{args.fail_if_missing_worker_run_pct}"
        )
    if failed_pct > float(args.fail_if_worker_failed_pct):
        fail_reasons.append(
            f"worker_failed_pct({failed_pct:.3f})>{args.fail_if_worker_failed_pct}"
        )
    if ok_missing_metrics_pct > float(args.fail_if_worker_ok_but_metrics_missing_pct):
        fail_reasons.append(
            f"worker_ok_but_missing_metrics_pct({ok_missing_metrics_pct:.3f})>{args.fail_if_worker_ok_but_metrics_missing_pct}"
        )

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {
            "where": "fits_header_core.imagetyp in ('LIGHT','')",
            "n_light": n_total,
            "n_workers_expected": len(EXPECTED_WORKERS),
            "n_pairs": n_pairs,
        },
        "summary": {
            "missing_worker_run": n_missing_run,
            "missing_worker_run_pct": round(missing_run_pct, 6),
            "worker_failed": n_failed,
            "worker_failed_pct": round(failed_pct, 6),
            "worker_ok_but_missing_metrics": n_ok_but_missing_metrics,
            "worker_ok_but_missing_metrics_pct": round(ok_missing_metrics_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_worker_run_pct": float(
                args.fail_if_missing_worker_run_pct
            ),
            "fail_if_worker_failed_pct": float(args.fail_if_worker_failed_pct),
            "fail_if_worker_ok_but_missing_metrics_missing_pct": float(
                args.fail_if_worker_ok_but_metrics_missing_pct
            ),
        },
        "fail_reasons": fail_reasons,
        "per_worker": [per_worker[w] for w in EXPECTED_WORKERS],
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "expected_workers": EXPECTED_WORKERS,
            "worker_to_metrics_table": WORKER_TO_METRICS_TABLE,
            "population_query": LIGHT_BASE_SQL.strip(),
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
                    "camera",
                    "object",
                    "filter",
                    "exptime",
                    "worker",
                    "issues",
                ]
            )
            for fr in failing:
                w.writerow(
                    [
                        fr.image_id,
                        fr.file_name,
                        fr.camera,
                        fr.object,
                        fr.filter,
                        fr.exptime,
                        fr.worker,
                        fr.issues,
                    ]
                )

    print(
        f"[test_{MODEL_NAME}] status={status} n_light={n_total} n_workers={len(EXPECTED_WORKERS)} "
        f"missing_run={n_missing_run} failed={n_failed} ok_but_missing_metrics={n_ok_but_missing_metrics}"
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
