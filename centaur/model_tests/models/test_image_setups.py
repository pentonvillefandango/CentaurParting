#!/usr/bin/env python3
# centaur/model_tests/models/test_image_setups.py
#
# Model test for image_setups (relational integrity + coverage).
#
# Deep checks:
#  - table exists
#  - PK integrity (nulls, duplicates)
#  - coverage vs images (missing setup rows)
#  - foreign-key integrity for all declared FKs (no orphan references)
#
# Outputs:
#   JSON results (always)
#   CSV failures (on FAIL or if --csv provided)
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


MODEL = "image_setups"


def _stamp_now() -> str:
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


def _resolve_run_dir(run_dir: str, model: str, stamp: str) -> Path:
    if run_dir.strip():
        base = Path(run_dir) / model
        base.mkdir(parents=True, exist_ok=True)
        return base
    base = Path("data") / "model_tests" / model / f"test_results_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _pk_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    # pk is 1..N (order for composite keys)
    pk_rows = [r for r in rows if int(r["pk"] or 0) > 0]
    pk_rows.sort(key=lambda rr: int(rr["pk"]))
    return [str(r["name"]) for r in pk_rows]


def _foreign_keys(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    """
    PRAGMA foreign_key_list returns one row per column in FK (composite supported).
    We'll return a grouped structure:
      [{"id":..., "ref_table":..., "from_cols":[...], "to_cols":[...]}]
    """
    rows = conn.execute(f"PRAGMA foreign_key_list({table});").fetchall()
    grouped: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        fid = int(r["id"])
        g = grouped.setdefault(
            fid,
            {
                "id": fid,
                "ref_table": str(r["table"]),
                "from_cols": [],
                "to_cols": [],
                "on_update": str(r["on_update"]),
                "on_delete": str(r["on_delete"]),
            },
        )
        g["from_cols"].append(str(r["from"]))
        g["to_cols"].append(str(r["to"]))
    return [grouped[k] for k in sorted(grouped.keys())]


@dataclass
class FailRow:
    image_id: int
    file_name: str
    issues: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: image_setups")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir", type=str, default="", help="Master run directory (optional)"
    )
    ap.add_argument(
        "--stamp", type=str, default="", help="Optional run stamp from master runner"
    )
    ap.add_argument("--out", type=str, default="", help="Output JSON path (optional)")
    ap.add_argument("--csv", type=str, default="", help="Failures CSV path (optional)")
    ap.add_argument(
        "--max-fail-rows", type=int, default=200, help="Max failing rows to capture"
    )
    ap.add_argument(
        "--fail-if-missing-pct",
        type=float,
        default=0.0,
        help="Fail if pct of images missing a setup row is > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-orphan-fk-pct",
        type=float,
        default=0.0,
        help="Fail if pct of setup rows with an orphan FK is > this (0..100). Default 0.",
    )

    args = ap.parse_args()

    db_path = Path(args.db)
    stamp = args.stamp.strip() or _stamp_now()
    run_base = _resolve_run_dir(args.run_dir, MODEL, stamp)

    out_json = (
        Path(args.out)
        if args.out.strip()
        else (run_base / f"test_{MODEL}_{stamp}.json")
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else (run_base / f"test_{MODEL}_failures_{stamp}.csv")
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = _connect(db_path)
    except Exception as e:
        print(f"[test_{MODEL}] ERROR: cannot open db: {e}")
        return 1

    schema_issues: List[str] = []
    if not _table_exists(conn, "images"):
        schema_issues.append("missing_table:images")
    if not _table_exists(conn, "image_setups"):
        schema_issues.append("missing_table:image_setups")

    if schema_issues:
        result = {
            "test": MODEL,
            "status": "ERROR",
            "db_path": str(db_path),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL}] wrote_json={out_json}")
        conn.close()
        return 1

    pk = _pk_cols(conn, "image_setups")
    if not pk:
        # not fatal, but it's unusual
        schema_issues.append("image_setups_missing_primary_key")

    n_images = conn.execute("SELECT COUNT(*) FROM images;").fetchone()[0]
    n_rows = conn.execute("SELECT COUNT(*) FROM image_setups;").fetchone()[0]

    # Coverage: images that have no setup row (assumes image_id is join column if present)
    cols = set(_cols(conn, "image_setups"))
    has_image_id = "image_id" in cols

    missing_image_ids: List[int] = []
    if has_image_id:
        missing_image_ids = [
            int(r["image_id"])
            for r in conn.execute(
                """
                SELECT i.image_id
                FROM images i
                LEFT JOIN image_setups s ON s.image_id=i.image_id
                WHERE s.image_id IS NULL
                ORDER BY i.image_id;
                """
            ).fetchall()
        ]

    n_missing = len(missing_image_ids)
    missing_pct = (100.0 * float(n_missing) / float(n_images)) if n_images else 0.0

    # PK checks
    null_pk = 0
    dup_pk = 0
    if pk:
        null_where = " OR ".join([f"{c} IS NULL" for c in pk])
        null_pk = conn.execute(
            f"SELECT COUNT(*) FROM image_setups WHERE {null_where};"
        ).fetchone()[0]

        if len(pk) == 1:
            dup_pk = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT {pk[0]} FROM image_setups GROUP BY {pk[0]} HAVING COUNT(*)>1);"
            ).fetchone()[0]
        else:
            cols_join = ", ".join(pk)
            dup_pk = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT {cols_join} FROM image_setups GROUP BY {cols_join} HAVING COUNT(*)>1);"
            ).fetchone()[0]

    # FK integrity
    fks = _foreign_keys(conn, "image_setups")
    fk_orphan_counts: Dict[str, int] = {}
    fk_total_orphan = 0

    for fk in fks:
        ref_table = fk["ref_table"]
        from_cols = fk["from_cols"]
        to_cols = fk["to_cols"]

        # Only check if referenced table exists
        if not _table_exists(conn, ref_table):
            fk_orphan_counts[f"fk_ref_table_missing:{ref_table}"] = (
                fk_orphan_counts.get(f"fk_ref_table_missing:{ref_table}", 0) + 1
            )
            continue

        # Build join condition s.from = r.to (supports composite)
        conds = []
        null_conds = []
        for fc, tc in zip(from_cols, to_cols):
            conds.append(f"s.{fc} = r.{tc}")
            null_conds.append(f"s.{fc} IS NULL")

        on = " AND ".join(conds)
        any_fk_null = " AND ".join(
            null_conds
        )  # all-null means "not set" -> ignore orphan check
        sql = f"""
        SELECT COUNT(*) AS n
        FROM image_setups s
        LEFT JOIN {ref_table} r
          ON {on}
        WHERE r.{to_cols[0]} IS NULL
          AND NOT ({any_fk_null});
        """
        n_orphan = int(conn.execute(sql).fetchone()["n"])
        key = f"orphan_fk:{','.join(from_cols)}->{ref_table}({','.join(to_cols)})"
        fk_orphan_counts[key] = n_orphan
        fk_total_orphan += n_orphan

    orphan_fk_pct = (100.0 * float(fk_total_orphan) / float(n_rows)) if n_rows else 0.0

    # Fail rows CSV/examples
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    # Only enumerate per-image failures for coverage (keeps runtime small)
    if has_image_id and missing_image_ids:
        for iid in missing_image_ids[: int(args.max_fail_rows)]:
            fn = conn.execute(
                "SELECT file_name FROM images WHERE image_id=?;", (iid,)
            ).fetchone()
            file_name = (
                str(fn["file_name"]) if fn and fn["file_name"] is not None else ""
            )
            failing.append(
                FailRow(
                    image_id=iid, file_name=file_name, issues="missing_image_setup_row"
                )
            )

        issue_counts["missing_image_setup_row"] = len(missing_image_ids)

    # PK issues
    if null_pk:
        issue_counts["null_primary_key"] = int(null_pk)
    if dup_pk:
        issue_counts["duplicate_primary_key"] = int(dup_pk)

    # FK issues
    for k, v in fk_orphan_counts.items():
        if v:
            issue_counts[k] = int(v)

    # Fail reasons
    fail_reasons: List[str] = []
    if n_images == 0:
        fail_reasons.append("no_images_found")

    if schema_issues:
        fail_reasons.extend(schema_issues)

    if null_pk > 0:
        fail_reasons.append(f"null_pk({null_pk})>0")
    if dup_pk > 0:
        fail_reasons.append(f"dup_pk({dup_pk})>0")

    if has_image_id and missing_pct > float(args.fail_if_missing_pct):
        fail_reasons.append(
            f"missing_image_setup_row_pct({missing_pct:.3f})>{args.fail_if_missing_pct}"
        )

    if orphan_fk_pct > float(args.fail_if_orphan_fk_pct):
        fail_reasons.append(
            f"orphan_fk_pct({orphan_fk_pct:.3f})>{args.fail_if_orphan_fk_pct}"
        )

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL,
        "status": status,
        "db_path": str(db_path),
        "schema": {
            "image_setups_cols": sorted(list(cols)),
            "primary_key_cols": pk,
            "foreign_keys": fks,
        },
        "population": {
            "n_images": int(n_images),
            "n_rows": int(n_rows),
            "has_image_id_col": bool(has_image_id),
            "missing_image_setup_rows": int(n_missing),
            "missing_image_setup_rows_pct": round(missing_pct, 6),
            "missing_image_ids": missing_image_ids[:500],
            "null_pk": int(null_pk),
            "dup_pk": int(dup_pk),
            "orphan_fk_total": int(fk_total_orphan),
            "orphan_fk_pct": round(orphan_fk_pct, 6),
        },
        "thresholds": {
            "fail_if_missing_pct": float(args.fail_if_missing_pct),
            "fail_if_orphan_fk_pct": float(args.fail_if_orphan_fk_pct),
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "max_fail_rows_captured": int(args.max_fail_rows),
            "output_dir": str(run_base),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "file_name", "issues"])
            for fr in failing:
                w.writerow([fr.image_id, fr.file_name, fr.issues])

    print(
        f"[test_{MODEL}] status={status} n_images={n_images} n_rows={n_rows} missing={n_missing} null_pk={null_pk} dup_pk={dup_pk} orphan_fk_total={fk_total_orphan}"
    )
    if fail_reasons:
        for rr in fail_reasons:
            print(f"  reason: {rr}")
    if issue_counts:
        top = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        for k, v in top:
            print(f"  issue::{k}={v}")

    print(f"[test_{MODEL}] wrote_json={out_json}")
    if write_csv:
        print(f"[test_{MODEL}] wrote_csv={out_csv}")

    conn.close()
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
