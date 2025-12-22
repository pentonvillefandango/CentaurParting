#!/usr/bin/env python3
# centaur/model_tests/models/test_fits_header_full.py
#
# Model test for fits_header_full (produced by fits_header_worker).
#
# fits_header_full may be:
#   - "wide" (instrume/imagetyp/filter/exptime/object columns), OR
#   - "payload" (JSON/text/blob) containing FITS header cards.
#
# This test is schema-adaptive and keeps deep checks:
#   - PK integrity, join coverage vs images
#   - Worker-run triage:
#       missing_worker_run
#       worker_failed
#       worker_ok_but_missing_header_full_row
#   - Payload deep checks:
#       payload present and parseable
#       payload interpreted into a key/value mapping
#       required keys present (case-insensitive), with policy:
#           - Always require INSTRUME, IMAGETYP
#           - Require EXPTIME (or accept EXPOSURE as fallback)
#           - Require OBJECT only for LIGHT
#           - Require FILTER only if header indicates a filter concept (FILTER/FILTER1/FILTER2/FWHEEL present)
#             Otherwise if OSC hint (BAYERPAT present), FILTER is not required (treat as "none")
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
from typing import Any, Dict, List, Optional, Tuple


MODEL = "fits_header_full"
WORKER = "fits_header_worker"

PAYLOAD_CANDIDATES = (
    "header_json",
    "header",
    "header_text",
    "cards_json",
    "cards",
    "raw_header",
    "raw",
    "blob",
    "data",
)

# Policy knobs (payload layout)
BASE_REQUIRED_KEYS = ["INSTRUME", "IMAGETYP"]
EXPTIME_KEYS = ["EXPTIME", "EXPOSURE"]  # accept either; EXPTIME preferred
LIGHT_ONLY_REQUIRED_KEYS = ["OBJECT"]

# FILTER logic:
# - if any of these exist anywhere, require FILTER to exist and be non-empty
FILTERISH_KEYS = ["FILTER", "FILTER1", "FILTER2", "FWHEEL"]
OSC_HINT_KEYS = ["BAYERPAT"]


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


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in set(_cols(conn, table))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        if v == float("inf") or v == float("-inf"):
            return None
        return v
    except Exception:
        return None


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


def _resolve_run_dir(run_dir: str, model: str, stamp: str) -> Path:
    """
    If --run-dir is provided (master runner), we write to:
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


@dataclass
class FailRow:
    image_id: int
    file_name: str
    issues: str


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    for t in ("images", "fits_header_full", "module_runs"):
        if not _table_exists(conn, t):
            issues.append(f"missing_table:{t}")

    if issues:
        return False, issues

    for col in ("image_id", "file_name", "status"):
        if not _has_col(conn, "images", col):
            issues.append(f"images_missing_col:{col}")

    if not _has_col(conn, "fits_header_full", "image_id"):
        issues.append("fits_header_full_missing_col:image_id")

    for col in ("image_id", "module_name", "status", "started_utc", "ended_utc"):
        if not _has_col(conn, "module_runs", col):
            issues.append(f"module_runs_missing_col:{col}")

    return (len(issues) == 0), issues


def _detect_layout(conn: sqlite3.Connection) -> Dict[str, Any]:
    cols = _cols(conn, "fits_header_full")
    cset = set(cols)
    ncols = len(cols)

    wide_required = {"instrume", "imagetyp", "filter", "exptime", "object"}
    if wide_required.issubset(cset):
        return {"layout": "wide", "payload_col": None, "ncols": ncols, "cols": cols}

    payload_col = None
    for c in PAYLOAD_CANDIDATES:
        if c in cset:
            payload_col = c
            break

    return {
        "layout": "payload",
        "payload_col": payload_col,
        "ncols": ncols,
        "cols": cols,
    }


def _json_get_ci(d: Dict[str, Any], key: str) -> Any:
    key_u = key.upper()
    for k, v in d.items():
        if str(k).upper() == key_u:
            return v
    return None


def _has_any_key_ci(d: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        if _json_get_ci(d, k) is not None:
            return True
    return False


def _parse_fits_card_line(s: str) -> Optional[Tuple[str, str]]:
    """
    Tiny FITS-ish parser:
      "EXPTIME = 300.0 / comment" -> ("EXPTIME", "300.0")
    """
    ss = s.strip()
    if "=" not in ss:
        return None
    left, right = ss.split("=", 1)
    key = left.strip()
    if not key:
        return None
    val = right.split("/", 1)[0].strip()
    if not val:
        return None
    return key, val


def _payload_to_kv_map(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Convert known payload shapes into a dict-like key/value mapping.

    Accepts:
      - dict: returned as-is
      - list of [key, value] pairs
      - list of {"key"/"keyword"/"name": ..., "value": ...}
      - list of strings like "KEY = VALUE"
    """
    if isinstance(obj, dict):
        return obj

    if isinstance(obj, list):
        out: Dict[str, Any] = {}
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k = str(item[0]).strip()
                if k:
                    out[k] = item[1]
                continue

            if isinstance(item, dict):
                k = None
                for kk in ("key", "keyword", "name", "card", "K", "KEY"):
                    if kk in item and item[kk] is not None:
                        k = str(item[kk]).strip()
                        break
                if k:
                    for vv in ("value", "val", "V", "VALUE"):
                        if vv in item:
                            out[k] = item[vv]
                            break
                    else:
                        out[k] = item
                continue

            if isinstance(item, str):
                parsed = _parse_fits_card_line(item)
                if parsed:
                    k, v = parsed
                    out[k] = v
                continue

        return out if out else None

    return None


def _is_blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Model test: fits_header_full")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir", type=str, default="", help="Master run directory (optional)"
    )
    ap.add_argument(
        "--stamp", type=str, default="", help="Optional run stamp from master runner"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="Output JSON path (optional; overrides run-dir behavior)",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="",
        help="Failures CSV path (optional; overrides run-dir behavior)",
    )
    ap.add_argument(
        "--max-fail-rows", type=int, default=200, help="Max failing rows to capture"
    )

    ap.add_argument("--fail-if-missing-row-pct", type=float, default=0.0)
    ap.add_argument("--fail-if-missing-worker-run-pct", type=float, default=0.0)
    ap.add_argument("--fail-if-worker-failed-pct", type=float, default=0.0)
    ap.add_argument("--fail-if-worker-ok-but-missing-row-pct", type=float, default=0.0)

    ap.add_argument("--fail-if-missing-payload-pct", type=float, default=0.0)
    ap.add_argument("--fail-if-missing-required-key-pct", type=float, default=0.0)

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

    ok_schema, schema_issues = _required_schema_checks(conn)
    if not ok_schema:
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

    layout_info = _detect_layout(conn)
    layout = layout_info["layout"]
    payload_col = layout_info["payload_col"]
    ncols = int(layout_info["ncols"])
    cols = list(layout_info["cols"])

    null_pk = conn.execute(
        "SELECT COUNT(*) FROM fits_header_full WHERE image_id IS NULL;"
    ).fetchone()[0]
    dup_pk = conn.execute(
        "SELECT COUNT(*) FROM (SELECT image_id FROM fits_header_full GROUP BY image_id HAVING COUNT(*)>1);"
    ).fetchone()[0]

    select_payload = (
        f", h.{payload_col} AS payload" if (layout == "payload" and payload_col) else ""
    )
    wide_select = (
        """
      , UPPER(TRIM(COALESCE(h.imagetyp,''))) AS imagetyp
      , TRIM(COALESCE(h.instrume,''))        AS instrume
      , UPPER(TRIM(COALESCE(h.filter,'')))   AS filter
      , TRIM(COALESCE(h.object,''))          AS object
      , CAST(h.exptime AS REAL)              AS exptime
    """
        if layout == "wide"
        else ""
    )

    join_sql = f"""
    SELECT
      i.image_id,
      i.file_name,
      i.status AS image_status,

      h.image_id AS h_image_id
      {wide_select}
      {select_payload}

      , mr.status      AS worker_status
      , mr.started_utc AS worker_started_utc
      , mr.ended_utc   AS worker_ended_utc
    FROM images i
    LEFT JOIN fits_header_full h
      ON h.image_id = i.image_id
    LEFT JOIN module_runs mr
      ON mr.rowid = (
        SELECT mr2.rowid
        FROM module_runs mr2
        WHERE mr2.image_id = i.image_id
          AND mr2.module_name = '{WORKER}'
        ORDER BY COALESCE(mr2.ended_utc, mr2.started_utc) DESC, mr2.rowid DESC
        LIMIT 1
      )
    ORDER BY i.image_id;
    """

    rows = [dict(r) for r in conn.execute(join_sql).fetchall()]
    n_total = len(rows)

    def pct(x: int) -> float:
        return (100.0 * float(x) / float(n_total)) if n_total else 0.0

    missing_row = 0
    missing_worker_run = 0
    worker_failed = 0
    worker_ok_but_missing_row = 0

    payload_empty = 0
    payload_json_parse_failed = 0
    payload_json_uninterpretable = 0
    required_key_missing = 0

    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []

    for r in rows:
        issues: List[str] = []

        ws = r.get("worker_status")
        ws_u = str(ws or "").upper() if ws is not None else None

        if ws is None:
            _add_issue(issues, "missing_worker_run")
        elif ws_u not in ("OK", "PASS"):
            _add_issue(issues, "worker_failed")

        if r.get("h_image_id") is None:
            _add_issue(issues, "missing_header_full_row")
            if ws_u in ("OK", "PASS"):
                _add_issue(issues, "worker_ok_but_missing_header_full_row")
        else:
            if layout == "wide":
                exptime = _safe_float(r.get("exptime"))
                if exptime is not None and exptime < 0:
                    _add_issue(issues, "exptime_negative")

                imagetyp = str(r.get("imagetyp") or "")
                if imagetyp == "LIGHT":
                    if not str(r.get("instrume") or "").strip():
                        _add_issue(issues, "instrume_empty_for_light")

                # Wide layout: keep original behavior (strict on schema presence, but not forcing filter/object here)
                # Note: wide layout implies those columns exist; their emptiness can be handled by downstream tests if needed.
            else:
                if not payload_col:
                    _add_issue(issues, "no_payload_column_found")
                else:
                    payload = r.get("payload")
                    payload_s = "" if payload is None else str(payload)

                    if not payload_s.strip():
                        _add_issue(issues, "payload_empty")
                    else:
                        parsed_obj: Any = None
                        try:
                            parsed_obj = json.loads(payload_s)
                        except Exception:
                            parsed_obj = None

                        if parsed_obj is None:
                            _add_issue(issues, "payload_json_parse_failed")
                        else:
                            kv = _payload_to_kv_map(parsed_obj)
                            if kv is None:
                                _add_issue(issues, "payload_json_uninterpretable")
                            else:
                                # Determine image type from payload (case-insensitive)
                                imagetyp_v = _json_get_ci(kv, "IMAGETYP")
                                imagetyp_u = str(imagetyp_v or "").strip().upper()

                                # Required keys per policy
                                missing_any = False

                                # Always required
                                for k in BASE_REQUIRED_KEYS:
                                    if _is_blank(_json_get_ci(kv, k)):
                                        missing_any = True

                                # Exposure time: accept EXPTIME or EXPOSURE (either non-blank)
                                exptime_present = False
                                for k in EXPTIME_KEYS:
                                    if not _is_blank(_json_get_ci(kv, k)):
                                        exptime_present = True
                                        break
                                if not exptime_present:
                                    _add_issue(issues, "missing_exptime_and_exposure")
                                    missing_any = True

                                # OBJECT required only for LIGHT
                                if imagetyp_u == "LIGHT":
                                    for k in LIGHT_ONLY_REQUIRED_KEYS:
                                        if _is_blank(_json_get_ci(kv, k)):
                                            missing_any = True

                                # FILTER policy:
                                # If header indicates filter concept exists -> require FILTER non-blank.
                                has_filterish = _has_any_key_ci(kv, FILTERISH_KEYS)
                                has_osc_hint = _has_any_key_ci(kv, OSC_HINT_KEYS)

                                if has_filterish:
                                    if _is_blank(_json_get_ci(kv, "FILTER")):
                                        missing_any = True
                                else:
                                    # No filterish keys at all:
                                    # - If OSC hint present (BAYERPAT), FILTER is not required (treat as "none")
                                    # - Otherwise, also do not require (unknown ecosystem)
                                    _ = has_osc_hint  # kept for readability / future warnings

                                if missing_any:
                                    _add_issue(
                                        issues, "required_key_missing_in_payload_json"
                                    )

        if "missing_header_full_row" in issues:
            missing_row += 1
        if "missing_worker_run" in issues:
            missing_worker_run += 1
        if "worker_failed" in issues:
            worker_failed += 1
        if "worker_ok_but_missing_header_full_row" in issues:
            worker_ok_but_missing_row += 1

        if "payload_empty" in issues:
            payload_empty += 1
        if "payload_json_parse_failed" in issues:
            payload_json_parse_failed += 1
        if "payload_json_uninterpretable" in issues:
            payload_json_uninterpretable += 1
        if "required_key_missing_in_payload_json" in issues:
            required_key_missing += 1

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1

            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        image_id=int(r["image_id"]),
                        file_name=str(r.get("file_name") or ""),
                        issues=";".join(issues),
                    )
                )

    fail_reasons: List[str] = []
    if n_total == 0:
        fail_reasons.append("no_images_found")

    if null_pk > 0:
        fail_reasons.append(f"null_pk_image_id({null_pk})>0")
    if dup_pk > 0:
        fail_reasons.append(f"duplicate_pk_image_id({dup_pk})>0")

    if pct(missing_row) > float(args.fail_if_missing_row_pct):
        fail_reasons.append(
            f"missing_header_full_row_pct({pct(missing_row):.3f})>{args.fail_if_missing_row_pct}"
        )

    if pct(missing_worker_run) > float(args.fail_if_missing_worker_run_pct):
        fail_reasons.append(
            f"missing_worker_run_pct({pct(missing_worker_run):.3f})>{args.fail_if_missing_worker_run_pct}"
        )

    if pct(worker_failed) > float(args.fail_if_worker_failed_pct):
        fail_reasons.append(
            f"worker_failed_pct({pct(worker_failed):.3f})>{args.fail_if_worker_failed_pct}"
        )

    if pct(worker_ok_but_missing_row) > float(
        args.fail_if_worker_ok_but_missing_row_pct
    ):
        fail_reasons.append(
            f"worker_ok_but_missing_header_full_row_pct({pct(worker_ok_but_missing_row):.3f})>{args.fail_if_worker_ok_but_missing_row_pct}"
        )

    if layout == "payload":
        if payload_col is None:
            fail_reasons.append("payload_layout_but_no_payload_column_found")

        if pct(payload_empty) > float(args.fail_if_missing_payload_pct):
            fail_reasons.append(
                f"payload_empty_pct({pct(payload_empty):.3f})>{args.fail_if_missing_payload_pct}"
            )

        if payload_json_parse_failed > 0:
            fail_reasons.append(
                f"payload_json_parse_failed({payload_json_parse_failed})"
            )

        if payload_json_uninterpretable > 0:
            fail_reasons.append(
                f"payload_json_uninterpretable({payload_json_uninterpretable})"
            )

        if pct(required_key_missing) > float(args.fail_if_missing_required_key_pct):
            fail_reasons.append(
                f"required_key_missing_in_payload_json_pct({pct(required_key_missing):.3f})>{args.fail_if_missing_required_key_pct}"
            )

    other_issue_total = sum(
        cnt
        for k, cnt in issue_counts.items()
        if k
        not in (
            "missing_worker_run",
            "worker_failed",
            "missing_header_full_row",
            "worker_ok_but_missing_header_full_row",
        )
    )
    if other_issue_total > 0 and not fail_reasons:
        fail_reasons.append(f"other_invariant_violations({other_issue_total})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL,
        "status": status,
        "db_path": str(db_path),
        "worker": WORKER,
        "layout": {
            "layout": layout,
            "ncols": ncols,
            "cols": cols,
            "payload_col": payload_col,
            "payload_candidates": list(PAYLOAD_CANDIDATES),
        },
        "population": {
            "n_total_images": n_total,
            "null_pk_image_id": int(null_pk),
            "duplicate_pk_image_id": int(dup_pk),
            "missing_header_full_row": int(missing_row),
            "missing_header_full_row_pct": round(pct(missing_row), 6),
            "missing_worker_run": int(missing_worker_run),
            "missing_worker_run_pct": round(pct(missing_worker_run), 6),
            "worker_failed": int(worker_failed),
            "worker_failed_pct": round(pct(worker_failed), 6),
            "worker_ok_but_missing_header_full_row": int(worker_ok_but_missing_row),
            "worker_ok_but_missing_header_full_row_pct": round(
                pct(worker_ok_but_missing_row), 6
            ),
            "payload_empty": int(payload_empty),
            "payload_empty_pct": round(pct(payload_empty), 6),
            "payload_json_parse_failed": int(payload_json_parse_failed),
            "payload_json_uninterpretable": int(payload_json_uninterpretable),
            "required_key_missing_in_payload_json": int(required_key_missing),
            "required_key_missing_in_payload_json_pct": round(
                pct(required_key_missing), 6
            ),
        },
        "thresholds": {
            "fail_if_missing_row_pct": float(args.fail_if_missing_row_pct),
            "fail_if_missing_worker_run_pct": float(
                args.fail_if_missing_worker_run_pct
            ),
            "fail_if_worker_failed_pct": float(args.fail_if_worker_failed_pct),
            "fail_if_worker_ok_but_missing_row_pct": float(
                args.fail_if_worker_ok_but_missing_row_pct
            ),
            "fail_if_missing_payload_pct": float(args.fail_if_missing_payload_pct),
            "fail_if_missing_required_key_pct": float(
                args.fail_if_missing_required_key_pct
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
            "join_sql": join_sql.strip(),
            "policy": {
                "base_required_keys": BASE_REQUIRED_KEYS,
                "exptime_keys_any_of": EXPTIME_KEYS,
                "light_only_required_keys": LIGHT_ONLY_REQUIRED_KEYS,
                "filterish_keys_trigger_required_filter": FILTERISH_KEYS,
                "osc_hint_keys": OSC_HINT_KEYS,
            },
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
        f"[test_{MODEL}] status={status} n_images={n_total} layout={layout} ncols={ncols} "
        f"missing_row={missing_row} missing_worker_run={missing_worker_run} worker_failed={worker_failed} worker_ok_but_missing={worker_ok_but_missing_row} "
        f"payload_empty={payload_empty} json_parse_failed={payload_json_parse_failed} required_key_missing={required_key_missing}"
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
