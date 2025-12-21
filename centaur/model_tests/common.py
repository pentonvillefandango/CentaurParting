from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return bool(r)


def columns(conn: sqlite3.Connection, table: str) -> List[str]:
    if not table_exists(conn, table):
        return []
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [str(r["name"]) for r in rows]


def col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in set(columns(conn, table))


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
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


def clamp01(v: Optional[float]) -> bool:
    if v is None:
        return True
    return 0.0 <= v <= 1.0


@dataclass
class TestResult:
    model: str
    passed: bool
    summary: str
    details: Dict[str, Any]


def write_result(out_path: Path, result: TestResult) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.model,
        "passed": bool(result.passed),
        "summary": result.summary,
        "details": result.details,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def ok(model: str, summary: str, **details: Any) -> TestResult:
    return TestResult(model=model, passed=True, summary=summary, details=details)


def fail(model: str, summary: str, **details: Any) -> TestResult:
    return TestResult(model=model, passed=False, summary=summary, details=details)


def sample_light_image_ids(conn: sqlite3.Connection, limit: int = 2000) -> List[int]:
    rows = conn.execute(
        """
        SELECT i.image_id
        FROM images i
        LEFT JOIN fits_header_core h USING(image_id)
        WHERE COALESCE(UPPER(TRIM(h.imagetyp)), '') IN ('LIGHT','')
        ORDER BY i.image_id DESC
        LIMIT ?;
        """,
        (int(limit),),
    ).fetchall()
    return [int(r["image_id"]) for r in rows]


def count_rows_in_ids(conn: sqlite3.Connection, table: str, ids: List[int]) -> int:
    if not ids:
        return 0
    q = ",".join(["?"] * len(ids))
    r = conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE image_id IN ({q});", tuple(ids)).fetchone()
    return int(r["n"] or 0)


def basic_table_health_check(
    conn: sqlite3.Connection,
    table: str,
    ids: List[int],
    coverage_min: float = 0.80,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Generic model test:
      - table exists
      - has image_id
      - coverage over LIGHT ids
      - if expected_fields/written_fields exist: check wf>0 when ef>0
      - if usable/reason exist: check usable=0 should have reason non-empty, usable=1 should not have reason='ok' weirdness
    """
    details: Dict[str, Any] = {}

    if not table_exists(conn, table):
        return False, {"error": "missing_table"}

    cols = set(columns(conn, table))
    details["columns"] = sorted(cols)

    if "image_id" not in cols:
        return False, {"error": "missing_column:image_id", "columns": sorted(cols)}

    n_lights = len(ids)
    n_rows = count_rows_in_ids(conn, table, ids)
    coverage = (n_rows / max(1, n_lights))
    details["n_lights_sampled"] = n_lights
    details["n_rows_present_for_sample"] = n_rows
    details["coverage"] = round(coverage, 4)

    if coverage < coverage_min:
        details["coverage_min_required"] = coverage_min
        return False, {"error": "low_coverage", **details}

    # sample rows for deeper checks
    sample_n = min(300, n_lights)
    sample_ids = ids[:sample_n]
    q = ",".join(["?"] * len(sample_ids))
    sample_rows = conn.execute(
        f"SELECT * FROM {table} WHERE image_id IN ({q}) ORDER BY image_id DESC;",
        tuple(sample_ids),
    ).fetchall()
    details["sample_rows_checked"] = len(sample_rows)

    # expected_fields/written_fields logic
    if "expected_fields" in cols and "written_fields" in cols:
        ef0_wf0 = 0
        efpos_wf0 = 0
        for r in sample_rows:
            ef = safe_int(r["expected_fields"]) or 0
            wf = safe_int(r["written_fields"]) or 0
            if ef == 0 and wf == 0:
                ef0_wf0 += 1
            if ef > 0 and wf == 0:
                efpos_wf0 += 1
        details["sample_expected0_written0"] = ef0_wf0
        details["sample_expected>0_written0"] = efpos_wf0
        if efpos_wf0 > 0:
            details["error"] = "expected_fields>0_but_written_fields=0_in_sample"
            return False, details

    # usable/reason sanity
    if "usable" in cols:
        bad_reason = 0
        ok_reason_weird = 0
        for r in sample_rows:
            u = safe_int(r["usable"])
            if u is None:
                continue
            reason = str(r["reason"] or "").strip() if "reason" in cols else ""
            if u == 0 and "reason" in cols and not reason:
                bad_reason += 1
            if u == 0 and reason.lower() == "ok":
                ok_reason_weird += 1
        details["sample_usable0_missing_reason"] = bad_reason
        details["sample_usable0_reason_ok_weird"] = ok_reason_weird
        if ok_reason_weird > 0:
            details["error"] = "usable0_reason_ok_weird_in_sample"
            return False, details

    return True, details
