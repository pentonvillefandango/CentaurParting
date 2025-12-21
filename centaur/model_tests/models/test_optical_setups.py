#!/usr/bin/env python3
# centaur/model_tests/models/test_optical_setups.py
#
# Model test for optical_setups (relational integrity + sanity).
#
# Deep checks:
#  - table exists
#  - PK integrity (nulls, duplicates; composite supported)
#  - FK integrity for declared FKs in optical_setups
#  - "referenced by image_setups" sanity (if image_setups references optical_setups)
#  - non-empty (configurable)
#
# Outputs:
#   JSON results (always)
#   CSV failures (on FAIL, or if --csv provided)
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
from typing import Any, Dict, List, Optional


MODEL = "optical_setups"


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
    """
    If --run-dir is provided (master runner), write to:
      <run_dir>/<model>/
    Otherwise (individual run):
      data/model_tests/<model>/test_results_<stamp>/
    """
    if run_dir.strip():
        base = Path(run_dir) / model
        base.mkdir(parents=True, exist_ok=True)
        return base

    base = Path("data") / "model_tests" / model / f"test_results_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _pk_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    pk_rows = [r for r in rows if int(r["pk"] or 0) > 0]
    pk_rows.sort(key=lambda rr: int(rr["pk"]))
    return [str(r["name"]) for r in pk_rows]


def _foreign_keys(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    """
    PRAGMA foreign_key_list returns one row per column in FK (composite supported).
    Return grouped:
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


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


@dataclass
class FailRow:
    pk: str
    issues: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: optical_setups")
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
        "--fail-if-empty",
        type=int,
        default=1,
        help="Fail if optical_setups has 0 rows (1=yes default, 0=no).",
    )
    ap.add_argument(
        "--fail-if-orphan-fk-pct",
        type=float,
        default=0.0,
        help="Fail if pct of optical_setups rows with orphan FK > this (0..100). Default 0.",
    )
    ap.add_argument(
        "--fail-if-image_setups_ref_missing_pct",
        type=float,
        default=0.0,
        help="If image_setups references optical_setups, fail if pct of referenced ids missing in optical_setups > this. Default 0.",
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
    if not _table_exists(conn, "optical_setups"):
        schema_issues.append("missing_table:optical_setups")

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

    cols = set(_cols(conn, "optical_setups"))
    pk = _pk_cols(conn, "optical_setups")
    fks = _foreign_keys(conn, "optical_setups")

    n_rows = int(conn.execute("SELECT COUNT(*) FROM optical_setups;").fetchone()[0])

    # PK checks
    null_pk = 0
    dup_pk = 0
    if pk:
        null_where = " OR ".join([f"{c} IS NULL" for c in pk])
        null_pk = int(
            conn.execute(
                f"SELECT COUNT(*) FROM optical_setups WHERE {null_where};"
            ).fetchone()[0]
        )

        if len(pk) == 1:
            dup_pk = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM (SELECT {pk[0]} FROM optical_setups GROUP BY {pk[0]} HAVING COUNT(*)>1);"
                ).fetchone()[0]
            )
        else:
            cols_join = ", ".join(pk)
            dup_pk = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM (SELECT {cols_join} FROM optical_setups GROUP BY {cols_join} HAVING COUNT(*)>1);"
                ).fetchone()[0]
            )
    else:
        schema_issues.append("optical_setups_missing_primary_key")

    # FK checks inside optical_setups
    fk_orphan_counts: Dict[str, int] = {}
    fk_total_orphan = 0
    for fk in fks:
        ref_table = fk["ref_table"]
        from_cols = fk["from_cols"]
        to_cols = fk["to_cols"]

        if not _table_exists(conn, ref_table):
            fk_orphan_counts[f"fk_ref_table_missing:{ref_table}"] = (
                fk_orphan_counts.get(f"fk_ref_table_missing:{ref_table}", 0) + 1
            )
            continue

        conds = []
        null_conds = []
        for fc, tc in zip(from_cols, to_cols):
            conds.append(f"s.{fc} = r.{tc}")
            null_conds.append(f"s.{fc} IS NULL")

        on = " AND ".join(conds)
        all_null = " AND ".join(null_conds)  # if all FK cols null => ignore as "unset"

        sql = f"""
        SELECT COUNT(*) AS n
        FROM optical_setups s
        LEFT JOIN {ref_table} r
          ON {on}
        WHERE r.{to_cols[0]} IS NULL
          AND NOT ({all_null});
        """
        n_orphan = int(conn.execute(sql).fetchone()["n"])
        key = f"orphan_fk:{','.join(from_cols)}->{ref_table}({','.join(to_cols)})"
        fk_orphan_counts[key] = n_orphan
        fk_total_orphan += n_orphan

    orphan_fk_pct = (100.0 * float(fk_total_orphan) / float(n_rows)) if n_rows else 0.0

    # Referenced-by-image_setups sanity (if an FK exists)
    image_setups_refs: List[Dict[str, Any]] = []
    image_setups_ref_missing = 0
    image_setups_ref_total_distinct = 0

    if _table_exists(conn, "image_setups"):
        # Find any FK from image_setups -> optical_setups
        img_fks = _foreign_keys(conn, "image_setups")
        image_setups_refs = [
            fk for fk in img_fks if fk["ref_table"] == "optical_setups"
        ]
        # If found, compute how many referenced ids are missing
        for fk in image_setups_refs:
            from_cols = fk["from_cols"]
            to_cols = fk["to_cols"]
            if len(from_cols) != len(to_cols) or not from_cols:
                continue

            # Count distinct referenced keys in image_setups (ignoring nulls)
            if len(from_cols) == 1:
                fc = from_cols[0]
                image_setups_ref_total_distinct = int(
                    conn.execute(
                        f"SELECT COUNT(DISTINCT {fc}) FROM image_setups WHERE {fc} IS NOT NULL;"
                    ).fetchone()[0]
                )
                image_setups_ref_missing = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) FROM (
                          SELECT DISTINCT s.{fc} AS k
                          FROM image_setups s
                          LEFT JOIN optical_setups o ON o.{to_cols[0]} = s.{fc}
                          WHERE s.{fc} IS NOT NULL
                            AND o.{to_cols[0]} IS NULL
                        );
                        """
                    ).fetchone()[0]
                )
            else:
                # Composite key: count distinct combos + missing combos
                fcs = ", ".join([f"s.{c}" for c in from_cols])
                tcs = " AND ".join(
                    [f"o.{tc} = s.{fc}" for fc, tc in zip(from_cols, to_cols)]
                )
                not_null = " AND ".join([f"s.{c} IS NOT NULL" for c in from_cols])

                image_setups_ref_total_distinct = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(from_cols)} FROM image_setups s WHERE {not_null});"
                    ).fetchone()[0]
                )
                image_setups_ref_missing = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*) FROM (
                          SELECT DISTINCT {fcs}
                          FROM image_setups s
                          LEFT JOIN optical_setups o ON {tcs}
                          WHERE {not_null}
                            AND o.rowid IS NULL
                        );
                        """
                    ).fetchone()[0]
                )

    image_setups_ref_missing_pct = (
        (
            100.0
            * float(image_setups_ref_missing)
            / float(image_setups_ref_total_distinct)
        )
        if image_setups_ref_total_distinct
        else 0.0
    )

    # Failures CSV (keep simple: only structural failures)
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    if n_rows == 0:
        issue_counts["no_rows"] = 1
    if schema_issues:
        for s in schema_issues:
            issue_counts[s] = issue_counts.get(s, 0) + 1
    if null_pk:
        issue_counts["null_primary_key"] = int(null_pk)
    if dup_pk:
        issue_counts["duplicate_primary_key"] = int(dup_pk)
    for k, v in fk_orphan_counts.items():
        if v:
            issue_counts[k] = int(v)
    if image_setups_ref_missing:
        issue_counts["image_setups_references_missing_in_optical_setups"] = int(
            image_setups_ref_missing
        )

    # Create a few failing rows (not all) if there are PK issues
    if pk and (null_pk or dup_pk) and n_rows:
        # Grab a sample of rows with null PK
        if null_pk:
            null_where = " OR ".join([f"{c} IS NULL" for c in pk])
            for r in conn.execute(
                f"SELECT * FROM optical_setups WHERE {null_where} LIMIT ?;",
                (int(args.max_fail_rows),),
            ):
                pkv = ", ".join([f"{c}={r[c]}" for c in pk])
                failing.append(FailRow(pk=pkv, issues="null_primary_key"))
        # Grab sample duplicates (single-column PK only, to keep it cheap)
        if dup_pk and len(pk) == 1:
            for r in conn.execute(
                f"""
                SELECT {pk[0]} AS k, COUNT(*) AS c
                FROM optical_setups
                GROUP BY {pk[0]}
                HAVING COUNT(*)>1
                LIMIT ?;
                """,
                (int(args.max_fail_rows),),
            ):
                failing.append(
                    FailRow(pk=f"{pk[0]}={r['k']}", issues="duplicate_primary_key")
                )

    # Fail reasons
    fail_reasons: List[str] = []
    if schema_issues:
        fail_reasons.extend(schema_issues)

    if int(args.fail_if_empty) == 1 and n_rows == 0:
        fail_reasons.append("no_optical_setups_found")

    if null_pk > 0:
        fail_reasons.append(f"null_pk({null_pk})>0")
    if dup_pk > 0:
        fail_reasons.append(f"dup_pk({dup_pk})>0")

    if orphan_fk_pct > float(args.fail_if_orphan_fk_pct):
        fail_reasons.append(
            f"orphan_fk_pct({orphan_fk_pct:.3f})>{args.fail_if_orphan_fk_pct}"
        )

    if image_setups_refs:
        if image_setups_ref_missing_pct > float(
            args.fail_if_image_setups_ref_missing_pct
        ):
            fail_reasons.append(
                f"image_setups_ref_missing_pct({image_setups_ref_missing_pct:.3f})>{args.fail_if_image_setups_ref_missing_pct}"
            )

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL,
        "status": status,
        "db_path": str(db_path),
        "schema": {
            "optical_setups_cols": sorted(list(cols)),
            "primary_key_cols": pk,
            "foreign_keys": fks,
            "image_setups_fk_to_optical_setups": image_setups_refs,
        },
        "population": {
            "n_rows": int(n_rows),
            "null_pk": int(null_pk),
            "dup_pk": int(dup_pk),
            "orphan_fk_total": int(fk_total_orphan),
            "orphan_fk_pct": round(orphan_fk_pct, 6),
            "image_setups_ref_total_distinct": int(image_setups_ref_total_distinct),
            "image_setups_ref_missing": int(image_setups_ref_missing),
            "image_setups_ref_missing_pct": round(image_setups_ref_missing_pct, 6),
        },
        "thresholds": {
            "fail_if_empty": int(args.fail_if_empty),
            "fail_if_orphan_fk_pct": float(args.fail_if_orphan_fk_pct),
            "fail_if_image_setups_ref_missing_pct": float(
                args.fail_if_image_setups_ref_missing_pct
            ),
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
            w.writerow(["pk", "issues"])
            for fr in failing:
                w.writerow([fr.pk, fr.issues])

    print(
        f"[test_{MODEL}] status={status} n_rows={n_rows} null_pk={null_pk} dup_pk={dup_pk} orphan_fk_total={fk_total_orphan} image_setups_ref_missing={image_setups_ref_missing}"
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
