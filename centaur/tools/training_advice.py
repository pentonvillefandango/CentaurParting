#!/usr/bin/env python3
"""
CentaurParting - Training Advice (stop-gap CLI)

Outputs (per target):
  - Recommended common exposure length (seconds)
  - Recommended SHO time fractions and ratios (vs Ha)
  - Diagnostics + always-on candidate table (including excluded candidates + reason)

Uses:
  - training_derived_metrics (required)
  - fits_header_core (required for target grouping)
  - dark_library_exposures (optional, if populated)

Key behaviour:
  - Groups results by target = COALESCE(NULLIF(TRIM(fits_header_core.object),''), '(unknown)')
  - Uses ALL usable frames for that target (no top-N trimming)
  - Always prints candidate table; excluded candidates show a reason.

Algorithm (deterministic):
  1) Candidate exposure set:
       - If --prefer-dark-library: ONLY exposures in dark_library_exposures
       - Else: exposures from dark_library_exposures if present, otherwise exposures seen in training for that target
  2) For each candidate exposure, compute per-filter efficiency:
       E = nebula_minus_bg_adu_s / sqrt(sky_ff_median_adu_s)
     using usable frames only (and per-target only).
  3) Discard exposures missing any required filters (default: SII, Ha, OIII)
  4) Apply safety gate:
       - linear_headroom_p99 must be >= min_linear_headroom (if column exists; NULL ignored)
  5) Score each exposure with:
       score = mean(E) * w_mean + min(E) * w_min
               - short_penalty * (min_exp / exp)
               - long_penalty  * (exp / max_exp)
  6) Choose exposure with highest score.
  7) At chosen exposure, compute filter time fractions:
       time ∝ 1 / E^2  (equal-SNR time allocation)
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_FILTERS = ["SII", "Ha", "OIII"]


@dataclass(frozen=True)
class ExposureEval:
    exptime_s: float
    status: str  # "OK" | "EXCLUDED"
    reason: str

    per_filter_E: Dict[str, float]
    per_filter_n: Dict[str, int]
    E_mean: float
    E_min: float
    score: float
    n_frames_used_total: int


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _compute_E(
    nebula_minus_bg_adu_s: float, sky_ff_median_adu_s: float
) -> Optional[float]:
    if nebula_minus_bg_adu_s is None or sky_ff_median_adu_s is None:
        return None
    if sky_ff_median_adu_s <= 0:
        return None
    if nebula_minus_bg_adu_s < 0:
        return None
    return nebula_minus_bg_adu_s / math.sqrt(sky_ff_median_adu_s)


def _fetch_targets(con: sqlite3.Connection) -> List[str]:
    if not _table_exists(con, "fits_header_core"):
        return []
    if not _table_exists(con, "training_derived_metrics"):
        return []

    rows = con.execute(
        """
        SELECT DISTINCT COALESCE(NULLIF(TRIM(h.object), ''), '(unknown)') AS target
        FROM training_derived_metrics tdm
        JOIN fits_header_core h ON h.image_id = tdm.image_id
        WHERE tdm.usable = 1
        ORDER BY target
        """
    ).fetchall()
    return [str(r[0]) for r in rows if r and r[0] is not None]


def _fetch_candidates_from_dark_library(con: sqlite3.Connection) -> List[float]:
    if not _table_exists(con, "dark_library_exposures"):
        return []
    rows = con.execute(
        "SELECT DISTINCT exptime_s FROM dark_library_exposures WHERE exptime_s IS NOT NULL"
    ).fetchall()
    out: List[float] = []
    for (v,) in rows:
        try:
            out.append(float(v))
        except Exception:
            pass
    return sorted(set(out))


def _fetch_candidates_from_training_for_target(
    con: sqlite3.Connection, target: str
) -> List[float]:
    rows = con.execute(
        """
        SELECT DISTINCT tdm.exptime_s
        FROM training_derived_metrics tdm
        JOIN fits_header_core h ON h.image_id = tdm.image_id
        WHERE tdm.usable = 1
          AND tdm.exptime_s IS NOT NULL
          AND COALESCE(NULLIF(TRIM(h.object), ''), '(unknown)') = ?
        """,
        (target,),
    ).fetchall()

    out: List[float] = []
    for (v,) in rows:
        try:
            out.append(float(v))
        except Exception:
            pass
    return sorted(set(out))


def evaluate_exposure_for_target(
    con: sqlite3.Connection,
    *,
    target: str,
    exptime_s: float,
    required_filters: Sequence[str],
    min_linear_headroom: float,
) -> Tuple[str, str, Dict[str, float], Dict[str, int], int]:
    """
    Returns:
      (status, reason, per_filter_E_avg, per_filter_n_used, n_used_total)

    status:
      - "OK"
      - "EXCLUDED"

    reason examples:
      - ok
      - no_frames_for_target_exposure
      - missing_required_filter:HA
      - all_frames_failed_headroom_gate
      - insufficient_valid_signal
    """
    has_lh = _has_column(con, "training_derived_metrics", "linear_headroom_p99")

    # Pull minimum fields (and linear_headroom_p99 if present)
    cols = [
        "tdm.filter",
        "tdm.nebula_minus_bg_adu_s",
        "tdm.sky_ff_median_adu_s",
    ]
    if has_lh:
        cols.append("tdm.linear_headroom_p99")

    sql = f"""
      SELECT {", ".join(cols)}
      FROM training_derived_metrics tdm
      JOIN fits_header_core h ON h.image_id = tdm.image_id
      WHERE tdm.usable = 1
        AND tdm.exptime_s = ?
        AND COALESCE(NULLIF(TRIM(h.object), ''), '(unknown)') = ?
        AND tdm.filter IS NOT NULL
        AND tdm.nebula_minus_bg_adu_s IS NOT NULL
        AND tdm.sky_ff_median_adu_s IS NOT NULL
    """
    rows = con.execute(sql, (exptime_s, target)).fetchall()
    if not rows:
        return (
            "EXCLUDED",
            "no_frames_for_target_exposure",
            {},
            {f.upper(): 0 for f in required_filters},
            0,
        )

    required_set = [f.strip().upper() for f in required_filters]
    per_filter_vals: Dict[str, List[float]] = {f: [] for f in required_set}
    per_filter_n: Dict[str, int] = {f: 0 for f in required_set}

    n_used = 0
    any_failed_headroom = False
    any_passed_headroom = False

    for row in rows:
        fu = str(row[0]).strip().upper()
        if fu not in per_filter_vals:
            continue

        neb = _safe_float(row[1])
        sky = _safe_float(row[2])
        if neb is None or sky is None:
            continue

        # Safety gate: linear headroom (if available)
        if has_lh:
            lh = _safe_float(row[3])
            if lh is not None and lh < min_linear_headroom:
                any_failed_headroom = True
                continue
            any_passed_headroom = True

        E = _compute_E(neb, sky)
        if E is None:
            continue

        per_filter_vals[fu].append(E)
        per_filter_n[fu] += 1
        n_used += 1

    if has_lh and (not any_passed_headroom) and any_failed_headroom:
        # We had frames, but they all got filtered out by headroom gate
        return ("EXCLUDED", "all_frames_failed_headroom_gate", {}, per_filter_n, 0)

    # Must have all required filters
    for f in required_set:
        if len(per_filter_vals[f]) == 0:
            return (
                "EXCLUDED",
                f"missing_required_filter:{f}",
                {},
                per_filter_n,
                n_used,
            )

    per_filter_E_avg: Dict[str, float] = {}
    for f in required_set:
        vals = per_filter_vals[f]
        per_filter_E_avg[f] = sum(vals) / float(len(vals))

    if not per_filter_E_avg:
        return ("EXCLUDED", "insufficient_valid_signal", {}, per_filter_n, n_used)

    return ("OK", "ok", per_filter_E_avg, per_filter_n, n_used)


def score_exposure(
    exptime_s: float,
    per_filter_E: Dict[str, float],
    min_exp: float,
    max_exp: float,
    w_mean: float,
    w_min: float,
    short_penalty: float,
    long_penalty: float,
) -> Tuple[float, float, float]:
    Es = list(per_filter_E.values())
    E_mean = sum(Es) / float(len(Es))
    E_min = min(Es)

    sp = short_penalty * (min_exp / exptime_s) if exptime_s > 0 else 1e9
    lp = long_penalty * (exptime_s / max_exp) if max_exp > 0 else 0.0

    score = (w_mean * E_mean) + (w_min * E_min) - sp - lp
    return score, E_mean, E_min


def compute_filter_time_fractions(per_filter_E: Dict[str, float]) -> Dict[str, float]:
    # Equal-SNR time allocation: t ∝ 1/E^2
    weights: Dict[str, float] = {}
    for f, E in per_filter_E.items():
        denom = E * E
        weights[f] = (1.0 / denom) if denom > 0 else 0.0

    s = sum(weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {f: 1.0 / n for f in weights}

    return {f: w / s for f, w in weights.items()}


def _print_candidate_table(
    cands: List[ExposureEval], required_filters: Sequence[str]
) -> None:
    # Compact, copy/paste friendly
    print("CANDIDATES (always shown):")
    for e in sorted(cands, key=lambda x: x.exptime_s):
        if e.status != "OK":
            print(f"  {e.exptime_s:>6.0f}s  EXCLUDED  reason={e.reason}")
            continue

        # Show quick summary + per-filter E + counts
        parts = [
            f"{e.exptime_s:>6.0f}s",
            f"score={e.score:.4f}",
            f"E_mean={e.E_mean:.4f}",
            f"E_min={e.E_min:.4f}",
            f"frames_used={e.n_frames_used_total}",
        ]
        print("  " + "  ".join(parts))
        # second line: per-filter E and frame counts
        pf = []
        for f in required_filters:
            fu = f.upper()
            pf.append(
                f"E[{fu}]={e.per_filter_E.get(fu, float('nan')):.4f} (n={e.per_filter_n.get(fu,0)})"
            )
        print("      " + "  ".join(pf))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="CentaurParting training advice (exposure + filter ratio), grouped by target"
    )
    ap.add_argument(
        "--db",
        required=True,
        help="Path to sqlite database (e.g. data/centaurparting.db)",
    )
    ap.add_argument(
        "--filters",
        default=",".join(DEFAULT_FILTERS),
        help="Comma-separated required filters (default: SII,Ha,OIII)",
    )
    ap.add_argument(
        "--min-linear-headroom",
        type=float,
        default=0.02,
        help="Reject frames with linear_headroom_p99 below this (if column exists). Default: 0.02",
    )
    ap.add_argument(
        "--w-mean",
        type=float,
        default=0.55,
        help="Weight for mean efficiency across filters (default: 0.55)",
    )
    ap.add_argument(
        "--w-min",
        type=float,
        default=0.45,
        help="Weight for worst-filter efficiency (default: 0.45)",
    )
    ap.add_argument(
        "--short-penalty",
        type=float,
        default=0.10,
        help="Penalty strength for very short exposures (default: 0.10)",
    )
    ap.add_argument(
        "--long-penalty",
        type=float,
        default=0.08,
        help="Penalty strength for very long exposures (default: 0.08)",
    )
    ap.add_argument(
        "--prefer-dark-library",
        action="store_true",
        help="If set, only consider exposures present in dark_library_exposures (must be populated).",
    )
    ap.add_argument(
        "--only-target",
        default="",
        help="Optional: run advice for one target name exactly (matches fits_header_core.object after trimming).",
    )

    args = ap.parse_args()
    required_filters = [s.strip().upper() for s in args.filters.split(",") if s.strip()]
    if not required_filters:
        print("ERROR: --filters produced an empty set", file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    try:
        con.row_factory = sqlite3.Row

        if not _table_exists(con, "training_derived_metrics"):
            print("ERROR: training_derived_metrics table not found.", file=sys.stderr)
            return 2
        if not _table_exists(con, "fits_header_core"):
            print(
                "ERROR: fits_header_core table not found (needed for per-target advice).",
                file=sys.stderr,
            )
            return 2

        targets = _fetch_targets(con)
        if not targets:
            print(
                "ERROR: No usable training rows found (training_derived_metrics.usable=1) to infer targets.",
                file=sys.stderr,
            )
            return 2

        if args.only_target.strip():
            ot = args.only_target.strip()
            targets = [t for t in targets if t == ot]
            if not targets:
                print(
                    f"ERROR: --only-target did not match any target with usable rows: {ot}",
                    file=sys.stderr,
                )
                return 2

        dark_candidates = _fetch_candidates_from_dark_library(con)

        print("CentaurParting Training Advice (per target)")
        print("=========================================")
        print(f"DB: {args.db}")
        print(f"Required filters: {', '.join(required_filters)}")
        if args.prefer_dark_library:
            print(
                "Candidate exposure source: dark_library_exposures ONLY (--prefer-dark-library)"
            )
        else:
            print(
                "Candidate exposure source: dark_library_exposures if present, else per-target training exposures"
            )
        print()

        any_success = False

        for target in targets:
            # Candidate set selection
            training_candidates = _fetch_candidates_from_training_for_target(
                con, target
            )

            if args.prefer_dark_library:
                if not dark_candidates:
                    print(f"TARGET: {target}")
                    print(
                        "  ERROR: --prefer-dark-library set but dark_library_exposures is missing/empty."
                    )
                    print()
                    continue
                candidates = dark_candidates
            else:
                # Use dark library if available; otherwise, use what exists for this target
                candidates = dark_candidates if dark_candidates else training_candidates

            if not candidates:
                print(f"TARGET: {target}")
                print(
                    "  ERROR: No candidate exposures found for this target (and dark library empty)."
                )
                print()
                continue

            min_exp = min(candidates)
            max_exp = max(candidates)

            evals: List[ExposureEval] = []

            for exp in candidates:
                status, reason, per_filter_E, per_filter_n, n_used = (
                    evaluate_exposure_for_target(
                        con,
                        target=target,
                        exptime_s=exp,
                        required_filters=required_filters,
                        min_linear_headroom=args.min_linear_headroom,
                    )
                )

                if status != "OK":
                    evals.append(
                        ExposureEval(
                            exptime_s=exp,
                            status="EXCLUDED",
                            reason=reason,
                            per_filter_E={},
                            per_filter_n=per_filter_n,
                            E_mean=0.0,
                            E_min=0.0,
                            score=float("-inf"),
                            n_frames_used_total=n_used,
                        )
                    )
                    continue

                score, E_mean, E_min = score_exposure(
                    exp,
                    per_filter_E,
                    min_exp=min_exp,
                    max_exp=max_exp,
                    w_mean=args.w_mean,
                    w_min=args.w_min,
                    short_penalty=args.short_penalty,
                    long_penalty=args.long_penalty,
                )

                evals.append(
                    ExposureEval(
                        exptime_s=exp,
                        status="OK",
                        reason="ok",
                        per_filter_E=per_filter_E,
                        per_filter_n=per_filter_n,
                        E_mean=E_mean,
                        E_min=E_min,
                        score=score,
                        n_frames_used_total=n_used,
                    )
                )

            ok_evals = [e for e in evals if e.status == "OK"]
            print(f"TARGET: {target}")

            if not ok_evals:
                print("  ERROR: No valid exposures (all candidates excluded).")
                _print_candidate_table(evals, required_filters)
                continue

            ok_evals.sort(key=lambda e: e.score, reverse=True)
            best = ok_evals[0]
            any_success = True

            fractions = compute_filter_time_fractions(best.per_filter_E)
            ref = "HA" if "HA" in fractions else list(fractions.keys())[0]
            ref_frac = fractions.get(ref, 1.0) or 1.0

            print(f"RECOMMENDED COMMON EXPOSURE: {best.exptime_s:.0f}s")
            print()

            print("RECOMMENDED FILTER TIME (equal-SNR):")
            for f in required_filters:
                fu = f.upper()
                tf = fractions.get(fu, 0.0)
                ratio = (tf / ref_frac) if ref_frac > 0 else 0.0
                print(f"  {fu:<4}  time_fraction={tf:.3f}   ratio_vs_{ref}={ratio:.2f}")
            print()

            ratios_compact = []
            for f in required_filters:
                fu = f.upper()
                ratio = (fractions.get(fu, 0.0) / ref_frac) if ref_frac > 0 else 0.0
                ratios_compact.append(f"{fu}:{ratio:.2f}")
            print(f"ONE-LINER: {best.exptime_s:.0f}s | " + " ".join(ratios_compact))
            print()

            print("DIAGNOSTICS (why this won):")
            print(
                f"  score={best.score:.4f}  E_mean={best.E_mean:.4f}  E_min={best.E_min:.4f}  frames_used_total={best.n_frames_used_total}"
            )
            for f in required_filters:
                fu = f.upper()
                n = best.per_filter_n.get(fu, 0)
                print(f"  E[{fu}]={best.per_filter_E[fu]:.4f}  (n={n})")
            print()

            _print_candidate_table(evals, required_filters)

        if not any_success:
            return 2
        return 0

    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
