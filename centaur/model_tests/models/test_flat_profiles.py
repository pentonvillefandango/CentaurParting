#!/usr/bin/env python3
# centaur/model_tests/models/test_flat_profiles.py
#
# Model test for flat_profiles.
# - Validates table presence + primary key integrity
# - PASS-with-warning if table is empty (unless --fail-if-empty 1)
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_flat_profiles.py --db data/centaurparting.db

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MODEL_NAME = "flat_profiles"
TABLE_NAME = "flat_profiles"


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


def _table_info(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return conn.execute(f"PRAGMA table_info({table});").fetchall()


def _pk_columns(table_info_rows: List[sqlite3.Row]) -> List[str]:
    # PRAGMA table_info: columns include (cid, name, type, notnull, dflt_value, pk)
    # pk > 0 indicates membership in primary key (and ordering for composite PK)
    pk_cols = [str(r["name"]) for r in table_info_rows if int(r["pk"] or 0) > 0]
    # Keep stable order by pk index if composite
    pk_cols_sorted = sorted(
        [
            (int(r["pk"] or 0), str(r["name"]))
            for r in table_info_rows
            if int(r["pk"] or 0) > 0
        ],
        key=lambda x: x[0],
    )
    return [name for _, name in pk_cols_sorted] if pk_cols_sorted else pk_cols


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class FailRow:
    pk_value: str
    issue: str


def _make_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    """
    Returns (run_dir, out_json, out_csv)
    - If --run-dir provided, write under <run_dir>/<model_name>/
    - Else write under data/model_tests/<model_name>/test_results_<stamp>/
    """
    stamp = _utc_stamp()

    if args.run_dir and str(args.run_dir).strip():
        run_dir = Path(args.run_dir) / MODEL_NAME
    else:
        run_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"

    _mkdir(run_dir)

    out_json = (
        Path(args.out)
        if args.out.strip()
        else run_dir / f"test_{MODEL_NAME}_{stamp}.json"
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else run_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv"
    )
    return run_dir, out_json, out_csv


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {MODEL_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="If set, write outputs under this directory (master runner uses this).",
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
        help="Max failing rows to capture in JSON/CSV",
    )
    ap.add_argument(
        "--fail-if-empty",
        type=int,
        default=0,
        help="If 1, FAIL when flat_profiles has 0 rows. Default 0 (PASS with warning).",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    run_dir, out_json, out_csv = _make_output_paths(args)

    try:
        conn = _connect(db_path)
    except Exception as e:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "error": f"cannot_open_db:{e}",
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        return 1

    # --- Schema checks
    schema_issues: List[str] = []
    if not _table_exists(conn, TABLE_NAME):
        schema_issues.append(f"missing_table:{TABLE_NAME}")

    if schema_issues:
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

    ti = _table_info(conn, TABLE_NAME)
    cols = [str(r["name"]) for r in ti]
    pk_cols = _pk_columns(ti)

    if not pk_cols:
        schema_issues.append("no_primary_key_defined")

    if schema_issues:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
            "table": TABLE_NAME,
            "columns": cols,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    # For this system, flat_profiles should have a simple PK, but we support composite too.
    # We will compute a string pk_value for CSV/debugging:
    pk_expr = " || '|' || ".join([f"COALESCE(CAST({c} AS TEXT),'')" for c in pk_cols])
    pk_expr = pk_expr if len(pk_cols) > 1 else f"CAST({pk_cols[0]} AS TEXT)"

    # --- Population
    n_rows = (
        _safe_int(
            conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME};").fetchone()["n"]
        )
        or 0
    )

    warnings: List[str] = []
    fail_reasons: List[str] = []
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    if n_rows == 0:
        if int(args.fail_if_empty) == 1:
            fail_reasons.append("no_flat_profiles_found")
        else:
            warnings.append("no_flat_profiles_found")

    # --- Deep integrity checks (only meaningful if rows exist)
    dup_pk = 0
    null_pk = 0

    if n_rows > 0:
        # NULL PK check: any PK col NULL means broken integrity
        null_where = " OR ".join([f"{c} IS NULL" for c in pk_cols])
        null_pk = (
            _safe_int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM {TABLE_NAME} WHERE {null_where};"
                ).fetchone()["n"]
            )
            or 0
        )
        if null_pk > 0:
            issue_counts["null_pk"] = null_pk
            fail_reasons.append(f"null_pk({null_pk})")

            # capture examples
            rows = conn.execute(
                f"SELECT {pk_expr} AS pk_value FROM {TABLE_NAME} WHERE {null_where} LIMIT ?;",
                (int(args.max_fail_rows),),
            ).fetchall()
            for r in rows:
                if len(failing) >= int(args.max_fail_rows):
                    break
                failing.append(FailRow(pk_value=str(r["pk_value"]), issue="null_pk"))

        # Duplicate PK check
        group_cols = ", ".join(pk_cols)
        dup_rows = conn.execute(
            f"""
            SELECT {pk_expr} AS pk_value, COUNT(*) AS c
            FROM {TABLE_NAME}
            GROUP BY {group_cols}
            HAVING COUNT(*) > 1
            ORDER BY c DESC
            LIMIT ?;
            """,
            (int(args.max_fail_rows),),
        ).fetchall()

        dup_pk = len(dup_rows)
        if dup_pk > 0:
            issue_counts["duplicate_pk_groups"] = dup_pk
            fail_reasons.append(f"duplicate_pk_groups({dup_pk})")
            for r in dup_rows:
                if len(failing) >= int(args.max_fail_rows):
                    break
                failing.append(
                    FailRow(
                        pk_value=str(r["pk_value"]),
                        issue=f"duplicate_pk(count={r['c']})",
                    )
                )

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "table": TABLE_NAME,
        "schema": {
            "columns": cols,
            "pk_columns": pk_cols,
        },
        "population": {
            "n_rows": n_rows,
        },
        "warnings": warnings,
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "run_dir": str(run_dir),
            "max_fail_rows_captured": int(args.max_fail_rows),
            "fail_if_empty": int(args.fail_if_empty),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pk_value", "issue"])
            for fr in failing:
                w.writerow([fr.pk_value, fr.issue])

    # Terminal summary
    print(
        f"[test_{MODEL_NAME}] status={status} n_rows={n_rows} dup_pk={dup_pk} null_pk={null_pk}"
    )
    for w in warnings:
        print(f"  warn: {w}")
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
