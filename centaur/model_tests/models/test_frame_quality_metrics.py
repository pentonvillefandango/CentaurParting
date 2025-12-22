#!/usr/bin/env python3
# centaur/model_tests/models/test_frame_quality_metrics.py
#
# Model test for frame_quality_metrics.
# - Uses a single LEFT JOIN base query (so we cannot "lose" rows in Python)
# - Validates schema + per-image invariants
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL (data invariant violation)
#     1 ERROR (test/system failure)
#
# Run:
#   python3 centaur/model_tests/models/test_frame_quality_metrics.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_frame_quality_metrics.py --db data/centaurparting.db --run-dir data/model_tests/test_results_<stamp>

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Allow running this file directly (script mode) without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from centaur.model_tests.common import (
    clamp01,
    col_exists,
    columns,
    connect,
    safe_float,
    safe_int,
    table_exists,
)

MODEL_NAME = "frame_quality_metrics"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


@dataclass
class FailRow:
    image_id: int
    file_name: str
    decision: str
    quality_score: int
    issues: str


# LIGHT frames (including imagetyp='') with LEFT JOIN to metrics
BASE_JOIN_SQL = """
SELECT
  i.image_id,
  i.file_name,
  i.status,
  UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp,

  m.image_id AS m_image_id,
  m.expected_fields,
  m.read_fields,
  m.written_fields,
  m.db_written_utc,
  m.parse_warnings,

  m.quality_score,
  m.decision,
  m.reason_mask,
  m.primary_reason,
  m.psf_score,
  m.bg_score,
  m.clip_score,
  m.usable,
  m.reason
FROM images i
JOIN fits_header_core h
  ON h.image_id = i.image_id
LEFT JOIN frame_quality_metrics m
  ON m.image_id = i.image_id
WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
ORDER BY i.image_id;
"""


def _required_schema_checks(conn) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    for t in ("images", "fits_header_core", "frame_quality_metrics"):
        if not table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    # Minimal columns relied on
    for col in ("image_id", "file_name", "status"):
        if not col_exists(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    for col in ("image_id", "imagetyp"):
        if not col_exists(conn, "fits_header_core", col):
            issues.append(f"fits_header_core_missing_col:{col}")

    for col in (
        "image_id",
        "expected_fields",
        "read_fields",
        "written_fields",
        "db_written_utc",
        "quality_score",
        "decision",
        "reason_mask",
        "primary_reason",
        "psf_score",
        "bg_score",
        "clip_score",
        "usable",
        "reason",
    ):
        if not col_exists(conn, "frame_quality_metrics", col):
            issues.append(f"frame_quality_metrics_missing_col:{col}")

    return (len(issues) == 0), issues


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    # Missing metrics row?
    if r.get("m_image_id") is None:
        _add_issue(issues, "missing_metrics_row")
        return issues

    ef = safe_int(r.get("expected_fields"))
    rf = safe_int(r.get("read_fields"))
    wf = safe_int(r.get("written_fields"))

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
    if ef is not None and ef > 0 and wf == 0:
        _add_issue(issues, "wrote_zero_fields")

    dbw = r.get("db_written_utc")
    if dbw is None or (isinstance(dbw, str) and not dbw.strip()):
        _add_issue(issues, "db_written_utc_missing")

    # Scores and decision
    qs = safe_int(r.get("quality_score"))
    if qs is None or not (0 <= qs <= 100):
        _add_issue(issues, "quality_score_out_of_range")

    dec = str(r.get("decision") or "").strip().upper()
    if dec not in ("KEEP", "WARN", "REJECT"):
        _add_issue(issues, "decision_invalid")

    for col in ("psf_score", "bg_score", "clip_score"):
        v = safe_int(r.get(col))
        if v is None or not (0 <= v <= 100):
            _add_issue(issues, f"{col}_out_of_range")

    rm = safe_int(r.get("reason_mask"))
    if rm is None or rm < 0:
        _add_issue(issues, "reason_mask_missing_or_negative")

    pr = str(r.get("primary_reason") or "").strip()
    if not pr:
        _add_issue(issues, "primary_reason_missing")

    # usable/reason sanity (aligns with your generic checks)
    u = safe_int(r.get("usable"))
    if u is None:
        _add_issue(issues, "usable_missing")
    else:
        reason = str(r.get("reason") or "").strip()
        if u == 0 and not reason:
            _add_issue(issues, "usable0_missing_reason")
        if u == 0 and reason.lower() == "ok":
            _add_issue(issues, "usable0_reason_ok_weird")

    # If decision says REJECT, should not claim usable=1
    if dec == "REJECT" and u == 1:
        _add_issue(issues, "reject_but_usable1")

    return issues


def _resolve_output_paths(
    *,
    stamp: str,
    out_arg: str,
    csv_arg: str,
    run_dir: str,
) -> Tuple[Path, Path, Path]:
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
    else:
        out_csv = out_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_dir, out_json, out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: frame_quality_metrics")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run output dir (if provided, writes to <run-dir>/frame_quality_metrics/ unless --out/--csv override)",
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
        conn = connect(db_path)
    except Exception as e:
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

    base_rows = conn.execute(BASE_JOIN_SQL).fetchall()
    rows = [dict(r) for r in base_rows]
    n_total = len(rows)

    missing_ids = [
        int(r["image_id"])
        for r in conn.execute(
            """
            SELECT i.image_id
            FROM images i
            JOIN fits_header_core h ON h.image_id=i.image_id
            LEFT JOIN frame_quality_metrics m ON m.image_id=i.image_id
            WHERE UPPER(TRIM(COALESCE(h.imagetyp,''))) IN ('LIGHT','')
              AND m.image_id IS NULL
            ORDER BY i.image_id;
            """
        ).fetchall()
    ]
    n_missing = len(missing_ids)
    missing_pct = (100.0 * float(n_missing) / float(n_total)) if n_total else 0.0

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues = _check_one_row(r)
        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1

            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        image_id=int(r["image_id"]),
                        file_name=str(r.get("file_name") or ""),
                        decision=str(r.get("decision") or ""),
                        quality_score=int(r.get("quality_score") or -1),
                        issues=";".join(issues),
                    )
                )

    fail_reasons: List[str] = []
    if n_total == 0:
        fail_reasons.append("no_light_frames_found")

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
        "output_dir": str(out_dir),
        "population": {
            "where": "imagetyp in ('LIGHT','') from fits_header_core",
            "n_total_frames": n_total,
            "n_missing_metrics_row": n_missing,
            "missing_metrics_row_pct": round(missing_pct, 6),
            "missing_image_ids": missing_ids[:500],
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
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
            w.writerow(["image_id", "file_name", "decision", "quality_score", "issues"])
            for fr in failing:
                w.writerow(
                    [
                        fr.image_id,
                        fr.file_name,
                        fr.decision,
                        fr.quality_score,
                        fr.issues,
                    ]
                )

    print(f"[test_{MODEL_NAME}] status={status} n_total={n_total} missing={n_missing}")
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
