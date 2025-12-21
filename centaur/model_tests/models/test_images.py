#!/usr/bin/env python3
# centaur/model_tests/models/test_images.py
#
# Core integrity test for images table.
# - Ensures images table exists + required columns
# - Validates:
#     * image_id present, positive, unique
#     * file_name present/nonblank
#     * status present/nonblank
#     * basic file_name sanity (extension heuristic)
#     * duplicates (file_name) reported (not always fatal, but surfaced)
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_images.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_images.py --db data/centaurparting.db --run-dir data/model_tests/test_results_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "images"


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


def _norm_trim(x: Any) -> str:
    return str(x or "").strip()


def _norm_lower(x: Any) -> str:
    return str(x or "").strip().lower()


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
    issues: str


BASE_SQL = """
SELECT
  image_id,
  file_name,
  status
FROM images
ORDER BY image_id;
"""


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if not _table_exists(conn, "images"):
        issues.append("missing_table:images")
        return False, issues

    for col in ("image_id", "file_name", "status"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    return (len(issues) == 0), issues


def _check_one_row(r: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    image_id = _safe_int(r.get("image_id"))
    file_name = _norm_trim(r.get("file_name"))
    status = _norm_trim(r.get("status"))

    if image_id is None:
        _add_issue(issues, "image_id_missing_or_nonint")
    elif image_id <= 0:
        _add_issue(issues, "image_id_nonpos")

    if not file_name:
        _add_issue(issues, "file_name_blank")
    else:
        # heuristic: FITS files usually end in .fit or .fits
        fn_lower = _norm_lower(file_name)
        if not (fn_lower.endswith(".fit") or fn_lower.endswith(".fits")):
            _add_issue(issues, "file_name_not_fit_or_fits")

    if not status:
        _add_issue(issues, "status_blank")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description="System test: images")
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
        default=500,
        help="Max failing rows to include in JSON/CSV",
    )

    ap.add_argument(
        "--fail-if-bad-extension-pct",
        type=float,
        default=100.0,
        help="Fail if file_name_not_fit_or_fits pct > this (0..100). Default 100 (never fail).",
    )
    ap.add_argument(
        "--fail-if-duplicate-filename-pct",
        type=float,
        default=100.0,
        help="Fail if duplicate file_name pct > this (0..100). Default 100 (never fail).",
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

    rows = [dict(r) for r in conn.execute(BASE_SQL).fetchall()]
    n_total = len(rows)

    # Deep integrity checks (SQL):
    dup_image_ids = [
        int(x["image_id"])
        for x in conn.execute(
            """
            SELECT image_id
            FROM images
            GROUP BY image_id
            HAVING COUNT(*) > 1
            ORDER BY image_id
            LIMIT 500;
            """
        ).fetchall()
    ]

    dup_file_names = [
        str(x["file_name"] or "")
        for x in conn.execute(
            """
            SELECT file_name
            FROM images
            WHERE TRIM(COALESCE(file_name,'')) <> ''
            GROUP BY file_name
            HAVING COUNT(*) > 1
            ORDER BY file_name
            LIMIT 500;
            """
        ).fetchall()
    ]

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    n_bad_ext = 0

    for r in rows:
        issues = _check_one_row(r)

        if "file_name_not_fit_or_fits" in issues:
            n_bad_ext += 1

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1

            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        image_id=int(_safe_int(r.get("image_id")) or 0),
                        file_name=str(r.get("file_name") or ""),
                        status=str(r.get("status") or ""),
                        issues=";".join(issues),
                    )
                )

    bad_ext_pct = (100.0 * n_bad_ext / n_total) if n_total else 0.0
    dup_fname_pct = (100.0 * len(dup_file_names) / n_total) if n_total else 0.0

    fail_reasons: List[str] = []

    if n_total == 0:
        fail_reasons.append("no_images_found")

    if dup_image_ids:
        fail_reasons.append(f"duplicate_image_id({len(dup_image_ids)})")

    if bad_ext_pct > float(args.fail_if_bad_extension_pct):
        fail_reasons.append(
            f"bad_extension_pct({bad_ext_pct:.3f})>{args.fail_if_bad_extension_pct}"
        )

    if dup_fname_pct > float(args.fail_if_duplicate_filename_pct):
        fail_reasons.append(
            f"duplicate_filename_pct({dup_fname_pct:.3f})>{args.fail_if_duplicate_filename_pct}"
        )

    # Force FAIL on these invariants if present (blank file_name/status, nonpos id)
    hard_fail_issues = (
        "image_id_missing_or_nonint",
        "image_id_nonpos",
        "file_name_blank",
        "status_blank",
    )
    hard_fail_total = sum(issue_counts.get(k, 0) for k in hard_fail_issues)
    if hard_fail_total > 0:
        fail_reasons.append(f"core_invariant_violations({hard_fail_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {
            "n_images": n_total,
        },
        "integrity": {
            "duplicate_image_ids": dup_image_ids[:500],
            "duplicate_file_names": dup_file_names[:500],
            "bad_extension_pct": round(bad_ext_pct, 6),
            "duplicate_file_name_pct": round(dup_fname_pct, 6),
        },
        "thresholds": {
            "fail_if_bad_extension_pct": float(args.fail_if_bad_extension_pct),
            "fail_if_duplicate_filename_pct": float(
                args.fail_if_duplicate_filename_pct
            ),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "base_query": BASE_SQL.strip(),
            "max_fail_rows_captured": int(args.max_fail_rows),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "file_name", "status", "issues"])
            for fr in failing:
                w.writerow([fr.image_id, fr.file_name, fr.status, fr.issues])

    print(
        f"[test_{MODEL_NAME}] status={status} n_images={n_total} dup_image_id={len(dup_image_ids)} "
        f"dup_file_name={len(dup_file_names)} bad_ext={n_bad_ext}"
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
