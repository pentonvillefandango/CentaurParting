#!/usr/bin/env python3
# centaur/model_tests/models/test_training_session_results.py
#
# Model test for training_session_results.
# Focus:
# - Latest result per training_session_id must be structurally valid
# - JSON fields parse and contain required keys
# - Recommended time fractions are sane (sum ~1, non-negative)
# - Exposure report and ratio fields are internally consistent
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


MODEL_NAME = "training_session_results"


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
    return [
        str(r["name"]) for r in conn.execute(f"PRAGMA table_info({table});").fetchall()
    ]


def _has_col(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return col in set(_cols(conn, table))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:
            return None
        if v in (float("inf"), float("-inf")):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _truthy_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _add_issue(issues: List[str], msg: str) -> None:
    if msg and msg not in issues:
        issues.append(msg)


def _resolve_output_dir(args_run_dir: str, stamp: str) -> Path:
    if args_run_dir.strip():
        out_dir = Path(args_run_dir) / MODEL_NAME
    else:
        out_dir = Path("data") / "model_tests" / MODEL_NAME / f"test_results_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@dataclass
class FailRow:
    training_session_id: int
    computed_utc: str
    status: str
    issues: str


def _required_schema_checks(conn: sqlite3.Connection) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not _table_exists(conn, MODEL_NAME):
        return False, [f"missing_table:{MODEL_NAME}"]

    # Minimal columns we need
    for col in (
        "training_session_id",
        "computed_utc",
        "status",
        "message",
        "recommended_exptime_s",
        "recommended_filters_json",
        "recommended_time_fraction_json",
        "recommended_ratio_vs_ha_json",
        "constraints_json",
        "conditions_json",
        "stats_json",
    ):
        if not _has_col(conn, MODEL_NAME, col):
            issues.append(f"{MODEL_NAME}_missing_col:{col}")

    return (len(issues) == 0), issues


def _json_load(s: Any) -> Optional[Any]:
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return None
    if not s.strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _check_time_fractions(tf: Any) -> List[str]:
    issues: List[str] = []
    if not isinstance(tf, dict) or not tf:
        return ["time_fractions_missing_or_not_object"]

    vals: List[float] = []
    for k, v in tf.items():
        if not _truthy_str(k):
            _add_issue(issues, "time_fraction_key_empty")
            continue
        fv = _safe_float(v)
        if fv is None:
            _add_issue(issues, f"time_fraction_not_float:{k}")
            continue
        if fv < 0:
            _add_issue(issues, f"time_fraction_negative:{k}")
        vals.append(float(fv))

    if not vals:
        _add_issue(issues, "time_fractions_all_invalid")
        return issues

    s = sum(vals)
    if s <= 0:
        _add_issue(issues, "time_fractions_sum_nonpos")
    # allow small rounding error
    if abs(s - 1.0) > 0.02:
        _add_issue(issues, f"time_fractions_sum_not_1(sum={s:.6f})")

    return issues


def _check_ratio_vs_ha(r: Any) -> List[str]:
    issues: List[str] = []
    if not isinstance(r, dict) or not r:
        return ["ratio_vs_ha_missing_or_not_object"]

    # values should be positive floats (ratios)
    for k, v in r.items():
        if not _truthy_str(k):
            _add_issue(issues, "ratio_key_empty")
            continue
        fv = _safe_float(v)
        if fv is None:
            _add_issue(issues, f"ratio_not_float:{k}")
            continue
        if fv <= 0:
            _add_issue(issues, f"ratio_nonpos:{k}")

    return issues


def _check_stats(stats: Any) -> List[str]:
    issues: List[str] = []
    if not isinstance(stats, dict):
        return ["stats_json_missing_or_not_object"]

    ratios = stats.get("ratios")
    if not isinstance(ratios, dict):
        _add_issue(issues, "stats_missing_ratios_object")
        return issues

    src = ratios.get("source_used")
    if not _truthy_str(src):
        _add_issue(issues, "ratios.source_used_missing")
    else:
        if str(src).strip().lower() not in ("best", "aggregate"):
            _add_issue(issues, f"ratios.source_used_unexpected:{src}")

    # exposure_report must exist and be object-like
    exp_rep = stats.get("exposure_report")
    if not isinstance(exp_rep, dict) or not exp_rep:
        _add_issue(issues, "stats.exposure_report_missing_or_empty")

    # aggregate exposures_used should exist (if aggregate block present)
    agg = ratios.get("aggregate")
    if isinstance(agg, dict):
        exps_used = agg.get("exposures_used")
        if exps_used is None:
            _add_issue(issues, "ratios.aggregate.exposures_used_missing")
        elif not isinstance(exps_used, list):
            _add_issue(issues, "ratios.aggregate.exposures_used_not_list")
        else:
            for x in exps_used:
                if _safe_int(x) is None:
                    _add_issue(issues, "ratios.aggregate.exposures_used_nonint")
                    break

    # If source_used is aggregate, ensure aggregate ratios exist
    if _truthy_str(src) and str(src).strip().lower() == "aggregate":
        if not isinstance(agg, dict):
            _add_issue(issues, "ratio_source_aggregate_but_missing_aggregate_block")
        else:
            if not isinstance(agg.get("time_fractions"), dict):
                _add_issue(issues, "aggregate.time_fractions_missing")
            if not isinstance(agg.get("ratio_vs_ha"), dict):
                _add_issue(issues, "aggregate.ratio_vs_ha_missing")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser(description=f"Model test: {MODEL_NAME}")
    ap.add_argument(
        "--db", type=str, default="data/centaurparting.db", help="Path to SQLite DB"
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Master run dir; outputs go into <run-dir>/<model>/",
    )
    ap.add_argument(
        "--out", type=str, default="", help="Output JSON path (optional override)"
    )
    ap.add_argument(
        "--csv", type=str, default="", help="Failures CSV path (optional override)"
    )
    ap.add_argument(
        "--max-fail-rows", type=int, default=200, help="Max failing rows to capture"
    )
    args = ap.parse_args()

    stamp = _utc_stamp()
    out_dir = _resolve_output_dir(args.run_dir, stamp)
    out_json = (
        Path(args.out)
        if args.out.strip()
        else (out_dir / f"test_{MODEL_NAME}_{stamp}.json")
    )
    out_csv = (
        Path(args.csv)
        if args.csv.strip()
        else (out_dir / f"test_{MODEL_NAME}_failures_{stamp}.csv")
    )

    try:
        conn = _connect(Path(args.db))
    except Exception as e:
        print(f"[test_{MODEL_NAME}] ERROR: cannot open db: {e}")
        return 1

    ok_schema, schema_issues = _required_schema_checks(conn)
    if not ok_schema:
        result = {
            "test": MODEL_NAME,
            "status": "ERROR",
            "db_path": str(args.db),
            "schema_issues": schema_issues,
        }
        out_json.write_text(json.dumps(result, indent=2))
        print(f"[test_{MODEL_NAME}] ERROR: schema issues: {schema_issues}")
        print(f"[test_{MODEL_NAME}] wrote_json={out_json}")
        conn.close()
        return 1

    # Latest row per training_session_id
    rows = conn.execute(
        """
        WITH latest AS (
          SELECT training_session_id, MAX(computed_utc) AS max_utc
          FROM training_session_results
          GROUP BY 1
        )
        SELECT r.*
        FROM training_session_results r
        JOIN latest l
          ON l.training_session_id = r.training_session_id
         AND l.max_utc = r.computed_utc
        ORDER BY r.training_session_id
        """
    ).fetchall()

    n_total = len(rows)
    issue_counts: Dict[str, int] = {}
    failing: List[FailRow] = []
    fail_reasons: List[str] = []

    for rr in rows:
        r = dict(rr)
        issues: List[str] = []

        sid = _safe_int(r.get("training_session_id")) or 0
        status = str(r.get("status") or "").strip().lower()
        computed_utc = str(r.get("computed_utc") or "")

        if not _truthy_str(computed_utc):
            _add_issue(issues, "computed_utc_missing")

        if status not in ("ok", "failed"):
            _add_issue(issues, f"status_unexpected:{status}")

        # status=ok implies recommended_exptime_s
        if status == "ok":
            if _safe_float(r.get("recommended_exptime_s")) is None:
                _add_issue(issues, "ok_but_recommended_exptime_missing")

        tf = _json_load(r.get("recommended_time_fraction_json"))
        ratio = _json_load(r.get("recommended_ratio_vs_ha_json"))
        stats = _json_load(r.get("stats_json"))

        # if status ok, tf + ratio should exist
        if status == "ok":
            issues += _check_time_fractions(tf)
            issues += _check_ratio_vs_ha(ratio)

        # stats_json should exist even if failed (your engine writes it)
        issues += _check_stats(stats)

        if issues:
            for it in issues:
                issue_counts[it] = issue_counts.get(it, 0) + 1
            if len(failing) < int(args.max_fail_rows):
                failing.append(
                    FailRow(
                        training_session_id=int(sid),
                        computed_utc=computed_utc,
                        status=status,
                        issues=";".join(issues),
                    )
                )

    if n_total == 0:
        fail_reasons.append("no_training_session_results_found")

    if issue_counts:
        fail_reasons.append(f"invariant_violations({sum(issue_counts.values())})")

    status = "PASS" if not fail_reasons else "FAIL"

    result: Dict[str, Any] = {
        "test": MODEL_NAME,
        "status": status,
        "db_path": str(args.db),
        "population": {
            "latest_row_per_training_session_id": True,
            "n_sessions_seen": n_total,
        },
        "fail_reasons": fail_reasons,
        "issue_counts": dict(
            sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "failing_examples": [fr.__dict__ for fr in failing],
    }

    out_json.write_text(json.dumps(result, indent=2))

    write_csv = (status == "FAIL") or bool(args.csv.strip())
    if write_csv:
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["training_session_id", "computed_utc", "status", "issues"])
            for fr in failing:
                w.writerow(
                    [fr.training_session_id, fr.computed_utc, fr.status, fr.issues]
                )

    print(f"[test_{MODEL_NAME}] status={status} n_sessions={n_total}")
    if fail_reasons:
        for rr in fail_reasons:
            print(f"  reason: {rr}")
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
