#!/usr/bin/env python3
# centaur/model_tests/models/test_flat_frame_links.py
#
# Model test for flat_frame_links.
# - Deep integrity checks for a link table:
#     * PK integrity: null PKs, duplicate PKs
#     * FK integrity: orphan references to images / flat_capture_sets / flat_profiles (if columns exist)
# - Output directory rules:
#     * If run directly: data/model_tests/flat_frame_links/test_results_<stamp>/
#     * If run by master: <run_dir>/flat_frame_links/
# - Outputs:
#     * JSON results (always)
#     * CSV failures (on FAIL, or if --csv provided)
# - Exit codes:
#     0 PASS
#     2 FAIL
#     1 ERROR
#
# Run:
#   python3 centaur/model_tests/models/test_flat_frame_links.py --db data/centaurparting.db

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MODEL_NAME = "flat_frame_links"
TABLE_NAME = "flat_frame_links"


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


def _pk_cols(conn: sqlite3.Connection, table: str) -> List[str]:
    """
    Return PK column names in PK order.
    PRAGMA table_info returns 'pk' as 1..N for composite keys.
    """
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    pk = [(int(r["pk"]), str(r["name"])) for r in rows if int(r["pk"] or 0) > 0]
    pk.sort(key=lambda t: t[0])
    return [name for _, name in pk]


def _has_col(cols: Sequence[str], col: str) -> bool:
    return col in set(cols)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


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


@dataclass
class FailRow:
    kind: str
    image_id: Optional[int]
    flat_capture_set_id: Optional[int]
    flat_profile_id: Optional[int]
    issues: str


def _pick_first_existing(cols: Sequence[str], options: Sequence[str]) -> Optional[str]:
    for c in options:
        if _has_col(cols, c):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: flat_frame_links")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )

    # master runner support
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="If set, write outputs under <run_dir>/flat_frame_links/",
    )
    # optional explicit stamp (master may want to force alignment)
    ap.add_argument(
        "--stamp",
        type=str,
        default="",
        help="Optional run stamp override (YYYYmmdd_HHMMSS)",
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
        "--max-fail-rows", type=int, default=500, help="Max failing rows to capture"
    )
    ap.add_argument(
        "--fail-if-empty",
        type=int,
        default=1,
        help="If 1, FAIL when no link rows are found. Default 1.",
    )

    args = ap.parse_args()

    stamp = args.stamp.strip() or _utc_stamp()
    db_path = Path(args.db)
    _, out_json, out_csv = _make_out_paths(args.run_dir, stamp, args.out, args.csv)

    try:
        conn = _connect(db_path)
    except Exception as e:
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        return 1

    schema_issues: List[str] = []
    warnings: List[str] = []

    for t in (TABLE_NAME, "images", "flat_capture_sets", "flat_profiles"):
        if not _table_exists(conn, t):
            schema_issues.append(f"missing_table:{t}")

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
        warnings.append("no_primary_key_detected")

    # Column name variants (schema drift tolerance)
    image_col = _pick_first_existing(cols, ["image_id"])
    capset_col = _pick_first_existing(cols, ["flat_capture_set_id", "capture_set_id"])
    profile_col = _pick_first_existing(cols, ["flat_profile_id", "profile_id"])

    if image_col is None:
        warnings.append(
            "flat_frame_links_missing_col:image_id (orphan_images check skipped)"
        )
    if capset_col is None:
        warnings.append(
            "flat_frame_links_missing_col:flat_capture_set_id/capture_set_id (orphan_capture_sets check skipped)"
        )
    if profile_col is None:
        warnings.append(
            "flat_frame_links_missing_col:flat_profile_id/profile_id (orphan_profiles check skipped)"
        )

    rows = conn.execute(f"SELECT * FROM {TABLE_NAME};").fetchall()
    n_rows = len(rows)

    # Empty policy
    if n_rows == 0:
        status = "FAIL" if int(args.fail_if_empty) == 1 else "PASS"
        fail_reasons = ["no_flat_frame_links_found"] if status == "FAIL" else []
        if status == "PASS":
            warnings.append("no_flat_frame_links_found")

        result = {
            "test": MODEL_NAME,
            "status": status,
            "db_path": str(db_path),
            "counts": {
                "n_rows": n_rows,
                "dup_pk": 0,
                "null_pk": 0,
                "orphan_images": 0,
                "orphan_capture_sets": 0,
                "orphan_profiles": 0,
            },
            "fail_reasons": fail_reasons,
            "warnings": warnings,
            "failing_examples": [],
            "notes": {"pk_cols": pk_cols, "columns": cols},
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] status={status} n_rows={n_rows} dup_pk=0 null_pk=0")
        for w in warnings[:10]:
            print(f"  warn: {w}")
        if fail_reasons:
            for r in fail_reasons:
                print(f"  reason: {r}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 0 if status == "PASS" else 2

    # ---- PK checks ----
    null_pk = 0
    dup_pk = 0

    if pk_cols:
        # null pk: any pk col is null
        where_null = " OR ".join([f"{c} IS NULL" for c in pk_cols])
        null_pk = conn.execute(
            f"SELECT COUNT(*) AS n FROM {TABLE_NAME} WHERE {where_null};"
        ).fetchone()["n"]

        # dup pk: group by pk cols having count>1
        grp = ", ".join(pk_cols)
        dup_pk = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM (
              SELECT {grp}, COUNT(*) AS c
              FROM {TABLE_NAME}
              GROUP BY {grp}
              HAVING COUNT(*) > 1
            );
            """
        ).fetchone()["n"]

    # ---- Orphan checks ----
    orphan_images = 0
    orphan_capture_sets = 0
    orphan_profiles = 0

    if image_col is not None:
        orphan_images = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {TABLE_NAME} l
            LEFT JOIN images i ON i.image_id = l.{image_col}
            WHERE l.{image_col} IS NOT NULL
              AND i.image_id IS NULL;
            """
        ).fetchone()["n"]

    if capset_col is not None:
        orphan_capture_sets = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {TABLE_NAME} l
            LEFT JOIN flat_capture_sets s ON s.flat_capture_set_id = l.{capset_col}
            WHERE l.{capset_col} IS NOT NULL
              AND s.flat_capture_set_id IS NULL;
            """
        ).fetchone()["n"]

    if profile_col is not None:
        orphan_profiles = conn.execute(
            f"""
            SELECT COUNT(*) AS n
            FROM {TABLE_NAME} l
            LEFT JOIN flat_profiles p ON p.flat_profile_id = l.{profile_col}
            WHERE l.{profile_col} IS NOT NULL
              AND p.flat_profile_id IS NULL;
            """
        ).fetchone()["n"]

    # ---- Build failure examples ----
    failing: List[FailRow] = []

    def add_examples(sql: str, kind: str, issue: str) -> None:
        nonlocal failing
        if len(failing) >= int(args.max_fail_rows):
            return
        ex = conn.execute(sql).fetchall()
        for r in ex:
            if len(failing) >= int(args.max_fail_rows):
                break
            failing.append(
                FailRow(
                    kind=kind,
                    image_id=_safe_int(r.get(image_col)) if image_col else None,
                    flat_capture_set_id=(
                        _safe_int(r.get(capset_col)) if capset_col else None
                    ),
                    flat_profile_id=(
                        _safe_int(r.get(profile_col)) if profile_col else None
                    ),
                    issues=issue,
                )
            )

    if pk_cols and null_pk > 0:
        where_null = " OR ".join([f"{c} IS NULL" for c in pk_cols])
        add_examples(
            f"SELECT * FROM {TABLE_NAME} WHERE {where_null} LIMIT 200;",
            "null_pk",
            "null_pk",
        )

    if pk_cols and dup_pk > 0:
        grp = ", ".join(pk_cols)
        add_examples(
            f"""
            SELECT l.*
            FROM {TABLE_NAME} l
            JOIN (
              SELECT {grp}
              FROM {TABLE_NAME}
              GROUP BY {grp}
              HAVING COUNT(*) > 1
            ) d
            ON {" AND ".join([f"l.{c}=d.{c}" for c in pk_cols])}
            LIMIT 200;
            """,
            "dup_pk",
            "dup_pk",
        )

    if image_col is not None and orphan_images > 0:
        add_examples(
            f"""
            SELECT l.*
            FROM {TABLE_NAME} l
            LEFT JOIN images i ON i.image_id = l.{image_col}
            WHERE l.{image_col} IS NOT NULL
              AND i.image_id IS NULL
            LIMIT 200;
            """,
            "orphan_images",
            "orphan_image_id",
        )

    if capset_col is not None and orphan_capture_sets > 0:
        add_examples(
            f"""
            SELECT l.*
            FROM {TABLE_NAME} l
            LEFT JOIN flat_capture_sets s ON s.flat_capture_set_id = l.{capset_col}
            WHERE l.{capset_col} IS NOT NULL
              AND s.flat_capture_set_id IS NULL
            LIMIT 200;
            """,
            "orphan_capture_sets",
            "orphan_flat_capture_set_id",
        )

    if profile_col is not None and orphan_profiles > 0:
        add_examples(
            f"""
            SELECT l.*
            FROM {TABLE_NAME} l
            LEFT JOIN flat_profiles p ON p.flat_profile_id = l.{profile_col}
            WHERE l.{profile_col} IS NOT NULL
              AND p.flat_profile_id IS NULL
            LIMIT 200;
            """,
            "orphan_profiles",
            "orphan_flat_profile_id",
        )

    # ---- Decide PASS/FAIL ----
    fail_reasons: List[str] = []
    if null_pk > 0:
        fail_reasons.append(f"null_pk({null_pk})")
    if dup_pk > 0:
        fail_reasons.append(f"dup_pk({dup_pk})")
    if orphan_images > 0:
        fail_reasons.append(f"orphan_images({orphan_images})")
    if orphan_capture_sets > 0:
        fail_reasons.append(f"orphan_capture_sets({orphan_capture_sets})")
    if orphan_profiles > 0:
        fail_reasons.append(f"orphan_profiles({orphan_profiles})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(db_path),
        "counts": {
            "n_rows": n_rows,
            "dup_pk": int(dup_pk),
            "null_pk": int(null_pk),
            "orphan_images": int(orphan_images),
            "orphan_capture_sets": int(orphan_capture_sets),
            "orphan_profiles": int(orphan_profiles),
        },
        "fail_reasons": fail_reasons,
        "warnings": warnings,
        "failing_examples": [fr.__dict__ for fr in failing],
        "notes": {
            "pk_cols": pk_cols,
            "columns": cols,
            "detected_columns": {
                "image_col": image_col,
                "flat_capture_set_col": capset_col,
                "flat_profile_col": profile_col,
            },
            "max_fail_rows_captured": int(args.max_fail_rows),
        },
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["kind", "image_id", "flat_capture_set_id", "flat_profile_id", "issues"]
            )
            for fr in failing:
                w.writerow(
                    [
                        fr.kind,
                        fr.image_id,
                        fr.flat_capture_set_id,
                        fr.flat_profile_id,
                        fr.issues,
                    ]
                )

    print(
        f"[test_{MODEL_NAME}] status={status} n_rows={n_rows} dup_pk={dup_pk} null_pk={null_pk} "
        f"orphan_images={orphan_images} orphan_capture_sets={orphan_capture_sets} orphan_profiles={orphan_profiles}"
    )
    if warnings:
        for w in warnings[:10]:
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
