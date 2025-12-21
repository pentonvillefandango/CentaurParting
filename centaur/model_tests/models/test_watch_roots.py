#!/usr/bin/env python3
# centaur/model_tests/models/test_watch_roots.py
#
# Model test for watch_roots (configuration/reference table).
#
# Checks:
#   - table exists, has columns
#   - table not empty
#   - primary key integrity (if PK declared):
#       * PK columns not NULL
#       * TEXT PK columns not blank
#       * no duplicate PK combinations
#   - root path integrity (if we can identify a path column):
#       * not NULL / not blank
#       * no duplicates case-insensitive after TRIM
#       * warn if not absolute-looking
#
# Outputs:
#   - JSON results (always)
#   - CSV failures (on FAIL, or if --csv provided)
#
# Exit codes:
#   0 PASS
#   2 FAIL
#   1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_watch_roots.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_watch_roots.py --db data/centaurparting.db --run-dir data/model_tests/test_results_YYYYMMDD_HHMMSS

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


TABLE = "watch_roots"
TEST_NAME = "watch_roots"


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


def _columns(info_rows: Sequence[sqlite3.Row]) -> List[str]:
    return [str(r["name"]) for r in info_rows]


def _pk_columns(info_rows: Sequence[sqlite3.Row]) -> List[str]:
    pk = [(int(r["pk"]), str(r["name"])) for r in info_rows if int(r["pk"] or 0) > 0]
    pk.sort(key=lambda t: t[0])
    return [name for _, name in pk]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_output_paths(
    *,
    model_name: str,
    stamp: str,
    run_dir: str,
    out_arg: str,
    csv_arg: str,
) -> Tuple[Path, Path, Path]:
    if run_dir.strip():
        model_out_dir = Path(run_dir) / model_name
    else:
        model_out_dir = (
            Path("data") / "model_tests" / model_name / f"test_results_{stamp}"
        )

    _ensure_dir(model_out_dir)

    json_path = (
        Path(out_arg)
        if out_arg.strip()
        else (model_out_dir / f"test_{model_name}_{stamp}.json")
    )
    csv_path = (
        Path(csv_arg)
        if csv_arg.strip()
        else (model_out_dir / f"test_{model_name}_failures_{stamp}.csv")
    )

    _ensure_dir(json_path.parent)
    _ensure_dir(csv_path.parent)

    return model_out_dir, json_path, csv_path


def _is_blank_text(v: Any) -> bool:
    return isinstance(v, str) and not v.strip()


def _looks_absolute_path(s: str) -> bool:
    # Unix/macOS absolute
    if s.startswith("/"):
        return True
    # Windows drive absolute like C:\ or C:/
    if re.match(r"^[A-Za-z]:[\\/]", s):
        return True
    # UNC path like \\server\share
    if s.startswith("\\\\"):
        return True
    return False


def _pick_path_column(cols: List[str]) -> Optional[str]:
    # Heuristics: common names first
    candidates = [
        "root_path",
        "watch_root",
        "path",
        "directory",
        "dir",
        "root",
    ]
    lower_to_actual = {c.lower(): c for c in cols}
    for want in candidates:
        if want in lower_to_actual:
            return lower_to_actual[want]
    return None


@dataclass
class FailRow:
    pk_or_row: str
    issue: str


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {TABLE}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # Master-run support:
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Root output directory for a master test run (each model writes to <run-dir>/<model>/...)",
    )

    # Optional explicit outputs:
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
        help="Max failing rows to capture in JSON/CSV",
    )

    args = ap.parse_args()

    stamp = _utc_stamp()
    _, out_json, out_csv = _resolve_output_paths(
        model_name=TABLE,
        stamp=stamp,
        run_dir=args.run_dir,
        out_arg=args.out,
        csv_arg=args.csv,
    )

    db_path = Path(args.db)

    try:
        conn = _connect(db_path)
    except Exception as e:
        result = {
            "test": TEST_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "error": f"cannot open db: {e}",
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{TABLE}] ERROR: cannot open db: {e}")
        print(f"[test_{TABLE}] wrote_json={out_json}")
        return 1

    schema_issues: List[str] = []
    if not _table_exists(conn, TABLE):
        schema_issues.append(f"missing_table:{TABLE}")

    if schema_issues:
        result = {
            "test": TEST_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{TABLE}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{TABLE}] wrote_json={out_json}")
        conn.close()
        return 1

    info = _table_info(conn, TABLE)
    cols = _columns(info)
    pk_cols = _pk_columns(info)

    if not cols:
        schema_issues.append("no_columns_found")

    if schema_issues:
        result = {
            "test": TEST_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
            "table_info": [dict(r) for r in info],
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{TABLE}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{TABLE}] wrote_json={out_json}")
        conn.close()
        return 1

    n_rows = conn.execute(f"SELECT COUNT(*) AS n FROM {TABLE};").fetchone()["n"]
    n_rows = int(n_rows or 0)

    fail_reasons: List[str] = []
    warn_reasons: List[str] = []
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    if n_rows == 0:
        fail_reasons.append("no_watch_roots_found")

    # --- PK checks (if PK exists)
    dup_pk_groups = 0
    null_pk_rows = 0
    blank_text_pk_rows = 0

    if pk_cols:
        # NULL PK
        null_where = " OR ".join([f"{c} IS NULL" for c in pk_cols])
        null_pk_rows = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM {TABLE} WHERE {null_where};"
            ).fetchone()["n"]
            or 0
        )
        if null_pk_rows > 0:
            issue_counts["null_primary_key"] = null_pk_rows
            sample = conn.execute(
                f"SELECT {', '.join(pk_cols)} FROM {TABLE} WHERE {null_where} LIMIT ?;",
                (int(args.max_fail_rows),),
            ).fetchall()
            for r in sample:
                pk_val = "|".join([str(r[c]) for c in pk_cols])
                failing.append(FailRow(pk_or_row=pk_val, issue="null_primary_key"))
            fail_reasons.append("null_primary_key_found")

        # Blank TEXT PK
        pk_decl_types = {str(r["name"]): str(r["type"] or "").upper() for r in info}
        text_pk_cols = [
            c
            for c in pk_cols
            if (
                "TEXT" in pk_decl_types.get(c, "") or "CHAR" in pk_decl_types.get(c, "")
            )
        ]
        if text_pk_cols:
            blank_where = " OR ".join([f"TRIM({c}) = ''" for c in text_pk_cols])
            blank_text_pk_rows = int(
                conn.execute(
                    f"SELECT COUNT(*) AS n FROM {TABLE} WHERE {blank_where};"
                ).fetchone()["n"]
                or 0
            )
            if blank_text_pk_rows > 0:
                issue_counts["blank_text_primary_key"] = blank_text_pk_rows
                sample = conn.execute(
                    f"SELECT {', '.join(pk_cols)} FROM {TABLE} WHERE {blank_where} LIMIT ?;",
                    (max(0, int(args.max_fail_rows) - len(failing)),),
                ).fetchall()
                for r in sample:
                    pk_val = "|".join([str(r[c]) for c in pk_cols])
                    failing.append(
                        FailRow(pk_or_row=pk_val, issue="blank_text_primary_key")
                    )
                fail_reasons.append("blank_text_primary_key_found")

        # Duplicate PK combos
        group_cols = ", ".join(pk_cols)
        dup_sql = f"""
        SELECT {group_cols}, COUNT(*) AS n
        FROM {TABLE}
        GROUP BY {group_cols}
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT ?;
        """
        dup_rows = conn.execute(dup_sql, (int(args.max_fail_rows),)).fetchall()
        dup_pk_groups = len(dup_rows)
        if dup_pk_groups > 0:
            issue_counts["duplicate_primary_key"] = dup_pk_groups
            for r in dup_rows[: max(0, int(args.max_fail_rows) - len(failing))]:
                pk_val = "|".join([str(r[c]) for c in pk_cols])
                failing.append(
                    FailRow(
                        pk_or_row=pk_val,
                        issue=f"duplicate_primary_key(n={int(r['n'])})",
                    )
                )
            fail_reasons.append("duplicate_primary_key_found")
    else:
        issue_counts["no_primary_key_declared"] = 1

    # --- Path column checks (best-effort, deep sanity)
    path_col = _pick_path_column(cols)
    path_checks = {
        "path_column": path_col,
        "null_path_rows": 0,
        "blank_path_rows": 0,
        "duplicate_path_groups_ci": 0,
        "non_absolute_path_rows": 0,
    }

    if path_col:
        # NULL
        null_path = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM {TABLE} WHERE {path_col} IS NULL;"
            ).fetchone()["n"]
            or 0
        )
        path_checks["null_path_rows"] = null_path
        if null_path > 0:
            issue_counts["root_path_null"] = null_path
            fail_reasons.append("root_path_null_found")
            sample = conn.execute(
                f"SELECT rowid, {path_col} FROM {TABLE} WHERE {path_col} IS NULL LIMIT ?;",
                (max(0, int(args.max_fail_rows) - len(failing)),),
            ).fetchall()
            for r in sample:
                failing.append(
                    FailRow(pk_or_row=f"rowid={r['rowid']}", issue="root_path_null")
                )

        # blank/whitespace
        blank_path = int(
            conn.execute(
                f"SELECT COUNT(*) AS n FROM {TABLE} WHERE TRIM({path_col}) = '';"
            ).fetchone()["n"]
            or 0
        )
        path_checks["blank_path_rows"] = blank_path
        if blank_path > 0:
            issue_counts["root_path_blank"] = blank_path
            fail_reasons.append("root_path_blank_found")
            sample = conn.execute(
                f"SELECT rowid, {path_col} FROM {TABLE} WHERE TRIM({path_col}) = '' LIMIT ?;",
                (max(0, int(args.max_fail_rows) - len(failing)),),
            ).fetchall()
            for r in sample:
                failing.append(
                    FailRow(pk_or_row=f"rowid={r['rowid']}", issue="root_path_blank")
                )

        # duplicates case-insensitive after TRIM
        dup_path_sql = f"""
        SELECT LOWER(TRIM({path_col})) AS p_norm, COUNT(*) AS n
        FROM {TABLE}
        WHERE {path_col} IS NOT NULL AND TRIM({path_col}) <> ''
        GROUP BY LOWER(TRIM({path_col}))
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT ?;
        """
        dup_path = conn.execute(dup_path_sql, (int(args.max_fail_rows),)).fetchall()
        path_checks["duplicate_path_groups_ci"] = len(dup_path)
        if dup_path:
            issue_counts["root_path_duplicate_case_insensitive"] = len(dup_path)
            fail_reasons.append("root_path_duplicate_case_insensitive_found")
            for r in (
                dup_path[: max(0, int(args.max_fail - rows) - len(failing))]
                if False
                else dup_path[: max(0, int(args.max_fail_rows) - len(failing))]
            ):
                failing.append(
                    FailRow(
                        pk_or_row=str(r["p_norm"]),
                        issue=f"root_path_duplicate_case_insensitive(n={int(r['n'])})",
                    )
                )

        # non-absolute (WARN)
        # Pull all non-null non-blank paths and check in Python
        paths = conn.execute(
            f"SELECT rowid, {path_col} AS p FROM {TABLE} WHERE {path_col} IS NOT NULL AND TRIM({path_col}) <> '';"
        ).fetchall()
        non_abs = 0
        for r in paths:
            p = str(r["p"])
            if not _looks_absolute_path(p):
                non_abs += 1
        path_checks["non_absolute_path_rows"] = non_abs
        if non_abs > 0:
            warn_reasons.append(f"non_absolute_watch_root_paths({non_abs})")
            issue_counts["root_path_non_absolute_warn"] = non_abs

    status = "PASS" if (not fail_reasons and n_rows > 0) else "FAIL"

    result: Dict[str, Any] = {
        "test": TEST_NAME,
        "status": status,
        "db_path": str(db_path),
        "table": TABLE,
        "row_counts": {"n_rows": n_rows},
        "schema": {
            "columns": cols,
            "pk_columns": pk_cols,
            "table_info": [dict(r) for r in info],
        },
        "pk_checks": {
            "dup_pk_groups": dup_pk_groups,
            "null_pk_rows": null_pk_rows,
            "blank_text_pk_rows": blank_text_pk_rows,
        },
        "path_checks": path_checks,
        "fail_reasons": fail_reasons,
        "warn_reasons": warn_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing[: int(args.max_fail_rows)]],
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pk_or_row", "issue"])
            for fr in failing[: int(args.max_fail_rows)]:
                w.writerow([fr.pk_or_row, fr.issue])

    # Terminal summary
    print(
        f"[test_{TABLE}] status={status} n_rows={n_rows} dup_pk_groups={dup_pk_groups} null_pk_rows={null_pk_rows}"
    )
    if fail_reasons:
        for r in fail_reasons:
            print(f"  reason: {r}")
    if warn_reasons:
        for r in warn_reasons:
            print(f"  warn: {r}")
    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  issue::{k}={v}")

    print(f"[test_{TABLE}] wrote_json={out_json}")
    if write_csv:
        print(f"[test_{TABLE}] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
