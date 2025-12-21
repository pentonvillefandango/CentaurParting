#!/usr/bin/env python3
# centaur/model_tests/models/test_flat_capture_sets.py
#
# Model test for flat_capture_sets.
# - Validates table presence + PK integrity (null/duplicate)
# - Validates declared foreign keys (orphan detection) using PRAGMA foreign_key_list
# - Output directory rules:
#     * If run directly: data/model_tests/flat_capture_sets/test_results_<stamp>/
#     * If run by master: <run_dir>/flat_capture_sets/
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_flat_capture_sets.py --db data/centaurparting.db
#   python3 centaur/model_tests/models/test_flat_capture_sets.py --db data/centaurparting.db --fail-if-empty 1

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MODEL_NAME = "flat_capture_sets"
TABLE_NAME = "flat_capture_sets"


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


def _fk_list(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    # columns: id, seq, table, from, to, on_update, on_delete, match
    return conn.execute(f"PRAGMA foreign_key_list({table});").fetchall()


def _cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [str(r["name"]) for r in _table_info(conn, table)]


def _has_col(cols: Sequence[str], col: str) -> bool:
    return col in set(cols)


def _pk_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    info = _table_info(conn, table)
    pks = [(int(r["pk"]), str(r["name"])) for r in info if int(r["pk"] or 0) > 0]
    pks.sort(key=lambda x: x[0])
    return [name for _, name in pks]


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


@dataclass
class FailRow:
    kind: str  # "null_pk" | "duplicate_pk" | "orphan_fk"
    detail: str


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: flat_capture_sets")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # master runner support
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="If set, write outputs under <run_dir>/flat_capture_sets/",
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

    # behavior
    ap.add_argument(
        "--fail-if-empty",
        type=int,
        default=0,
        help="If 1, FAIL when flat_capture_sets has 0 rows. Default 0 (PASS with warning).",
    )
    ap.add_argument(
        "--max-fail-rows",
        type=int,
        default=200,
        help="Max failing rows to include in JSON/CSV",
    )

    args = ap.parse_args()

    stamp = _utc_stamp()
    db_path = Path(args.db)
    _, out_json, out_csv = _make_out_paths(args.run_dir, stamp, args.out, args.csv)

    try:
        conn = _connect(db_path)
    except Exception as e:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": [f"cannot_open_db:{e}"],
            "warnings": [],
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        return 1

    schema_issues: List[str] = []
    warnings: List[str] = []

    if not _table_exists(conn, TABLE_NAME):
        schema_issues.append(f"missing_table:{TABLE_NAME}")

    if schema_issues:
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

    cols = _cols(conn, TABLE_NAME)
    pk_cols = _pk_cols(conn, TABLE_NAME)

    if not pk_cols:
        schema_issues.append("no_primary_key_defined")

    if schema_issues:
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

    n_rows = (
        _safe_int(conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0])
        or 0
    )

    if n_rows == 0:
        msg = "no_flat_capture_sets_found"
        if int(args.fail_if_empty) == 1:
            status = "FAIL"
            fail_reasons = [msg]
        else:
            status = "PASS"
            fail_reasons = []
            warnings.append(msg)

        result = {
            "test": MODEL_NAME,
            "status": status,
            "db_path": str(db_path),
            "population": {"table": TABLE_NAME, "n_rows": n_rows},
            "pk_columns": pk_cols,
            "fail_reasons": fail_reasons,
            "warnings": warnings,
            "issue_counts": {},
            "fk_orphan_counts": {},
            "failing_examples": [],
            "notes": {"max_fail_rows_captured": int(args.max_fail_rows)},
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(
            f"[test_{MODEL_NAME}] status={status} n_rows={n_rows} dup_pk=0 orphan_fk=0"
        )
        for w in warnings:
            print(f"  warn: {w}")
        if fail_reasons:
            for r in fail_reasons:
                print(f"  reason: {r}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 0 if status == "PASS" else 2

    # Deep checks
    issue_counts: Dict[str, int] = {}
    fk_orphan_counts: Dict[str, int] = {}
    failing: List[FailRow] = []
    max_examples = int(args.max_fail_rows)

    # Null PKs
    where_null_parts = [f"{c} IS NULL" for c in pk_cols]
    null_pk_count = (
        _safe_int(
            conn.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE "
                + " OR ".join(where_null_parts)
                + ";"
            ).fetchone()[0]
        )
        or 0
    )
    if null_pk_count > 0:
        issue_counts["null_pk"] = null_pk_count

        rows = conn.execute(
            f"SELECT {', '.join(pk_cols)} FROM {TABLE_NAME} WHERE "
            + " OR ".join(where_null_parts)
            + f" LIMIT {max_examples};"
        ).fetchall()
        for r in rows:
            pk_val = "|".join(_safe_str(r[c]) for c in pk_cols)
            failing.append(FailRow(kind="null_pk", detail=pk_val))

    # Duplicate PKs
    pk_expr = ", ".join(pk_cols)
    dup_rows = conn.execute(
        f"""
        SELECT {pk_expr}, COUNT(*) AS n
        FROM {TABLE_NAME}
        GROUP BY {pk_expr}
        HAVING COUNT(*) > 1
        ORDER BY n DESC
        LIMIT 500;
        """
    ).fetchall()
    dup_pk_count = len(dup_rows)
    if dup_pk_count > 0:
        issue_counts["duplicate_pk"] = dup_pk_count
        for r in dup_rows[: max(0, max_examples - len(failing))]:
            pk_val = "|".join(_safe_str(r[c]) for c in pk_cols)
            failing.append(FailRow(kind="duplicate_pk", detail=pk_val))

    # FK orphan checks (auto-discovered)
    fks = _fk_list(conn, TABLE_NAME)

    # Group by FK id (composite foreign keys have multiple rows with same id)
    by_id: Dict[int, List[sqlite3.Row]] = {}
    for fk in fks:
        by_id.setdefault(int(fk["id"]), []).append(fk)

    # For each FK constraint, build a LEFT JOIN orphan query
    for fk_id, rows_fk in sorted(by_id.items(), key=lambda kv: kv[0]):
        parent_table = str(rows_fk[0]["table"])
        if not parent_table:
            continue

        if not _table_exists(conn, parent_table):
            issue_counts[f"fk_parent_table_missing:{parent_table}"] = (
                issue_counts.get(f"fk_parent_table_missing:{parent_table}", 0) + 1
            )
            continue

        # Build join clauses for composite FK
        # child.from_col = parent.to_col for each seq
        rows_fk_sorted = sorted(rows_fk, key=lambda r: int(r["seq"]))
        join_conds = []
        child_null_conds = []
        label_parts = []
        for fkrow in rows_fk_sorted:
            c_from = str(fkrow["from"])
            p_to = str(fkrow["to"]) if fkrow["to"] is not None else "rowid"
            join_conds.append(f"c.{c_from} = p.{p_to}")
            child_null_conds.append(f"c.{c_from} IS NULL")
            label_parts.append(f"{c_from}->{parent_table}.{p_to}")

        label = f"fk{fk_id}:" + ",".join(label_parts)

        # We only count orphans where the child FK value is NOT NULL (NULL means "not linked", not an orphan)
        where_child_has_value = " AND ".join([f"NOT ({x})" for x in child_null_conds])

        orphan_sql = f"""
        SELECT COUNT(*) AS n
        FROM {TABLE_NAME} c
        LEFT JOIN {parent_table} p
          ON {" AND ".join(join_conds)}
        WHERE ({where_child_has_value})
          AND p.rowid IS NULL;
        """
        orphan_n = _safe_int(conn.execute(orphan_sql).fetchone()["n"]) or 0
        fk_orphan_counts[label] = orphan_n
        if orphan_n > 0:
            issue_counts["orphan_fk"] = issue_counts.get("orphan_fk", 0) + orphan_n

            # capture a few examples
            if len(failing) < max_examples:
                # show child pk + fk columns
                child_fk_cols = [str(r["from"]) for r in rows_fk_sorted]
                sel_cols = list(
                    dict.fromkeys(pk_cols + child_fk_cols)
                )  # preserve order
                ex_rows = conn.execute(
                    f"""
                    SELECT {", ".join("c."+c for c in sel_cols)}
                    FROM {TABLE_NAME} c
                    LEFT JOIN {parent_table} p
                      ON {" AND ".join(join_conds)}
                    WHERE ({where_child_has_value})
                      AND p.rowid IS NULL
                    LIMIT {max_examples - len(failing)};
                    """
                ).fetchall()
                for er in ex_rows:
                    detail = "|".join(f"{c}={_safe_str(er[c])}" for c in sel_cols)
                    failing.append(
                        FailRow(kind="orphan_fk", detail=f"{label} {detail}")
                    )

    fail_reasons: List[str] = []
    if null_pk_count > 0:
        fail_reasons.append(f"null_pk({null_pk_count})")
    if dup_pk_count > 0:
        fail_reasons.append(f"duplicate_pk({dup_pk_count})")
    if issue_counts.get("orphan_fk", 0) > 0:
        fail_reasons.append(f"orphan_fk({issue_counts['orphan_fk']})")

    # Any other invariant issues (like missing parent table) are real
    other_issue_total = sum(
        v
        for k, v in issue_counts.items()
        if k not in ("null_pk", "duplicate_pk", "orphan_fk")
    )
    if other_issue_total > 0:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "population": {"table": TABLE_NAME, "n_rows": n_rows},
        "pk_columns": pk_cols,
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "fk_orphan_counts": fk_orphan_counts,
        "failing_examples": [fr.__dict__ for fr in failing[:max_examples]],
        "notes": {"max_fail_rows_captured": max_examples},
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "detail"])
            for fr in failing[:max_examples]:
                w.writerow([fr.kind, fr.detail])

    orphan_fk_total = issue_counts.get("orphan_fk", 0)
    print(
        f"[test_{MODEL_NAME}] status={status} n_rows={n_rows} dup_pk={dup_pk_count} orphan_fk={orphan_fk_total}"
    )
    for w in warnings:
        print(f"  warn: {w}")
    if fail_reasons:
        for r in fail_reasons:
            print(f"  reason: {r}")

    print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
    if write_csv:
        print(f"[test_{MODEL_NAME}] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
