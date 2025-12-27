# centaur/training_session_engine.py
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any, Iterable

from centaur.logging import utc_now

DEFAULT_FILTERS = ["SII", "HA", "OIII"]


@dataclass(frozen=True)
class ExposureEval:
    exptime_s: float
    per_filter_E: Dict[str, float]
    E_mean: float
    E_min: float
    score: float
    n_frames_used: int
    min_linear_headroom_p99: Optional[float]


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _has_column(con: sqlite3.Connection, table: str, col: str) -> bool:
    rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == col for r in rows)


def _norm_target(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().strip("'\"")
    t = re.sub(r"\s+", " ", t)
    return t.lower() if t else None


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


def _safe_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
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


def fetch_training_session(con: sqlite3.Connection, sid: int) -> sqlite3.Row:
    row = con.execute(
        "SELECT * FROM training_sessions WHERE training_session_id=?",
        (sid,),
    ).fetchone()
    if row is None:
        raise ValueError(f"training_session_id {sid} not found")
    return row


def parse_params_json(sess: sqlite3.Row) -> Dict[str, Any]:
    raw = sess["params_json"]
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _infer_dark_profile_id(con: sqlite3.Connection, sid: int) -> Optional[int]:
    if _has_column(con, "training_sessions", "dark_profile_id"):
        row = con.execute(
            "SELECT dark_profile_id FROM training_sessions WHERE training_session_id=?",
            (sid,),
        ).fetchone()
        if row and row[0] is not None:
            return _safe_int(row[0])
    return None


def _candidate_exposures_for_session(con: sqlite3.Connection, sid: int) -> List[float]:
    rows = con.execute(
        """
        SELECT DISTINCT tsf.exptime_s
        FROM training_session_frames tsf
        WHERE tsf.training_session_id=?
          AND tsf.exptime_s IS NOT NULL
        """,
        (sid,),
    ).fetchall()
    out: List[float] = []
    for (v,) in rows:
        try:
            out.append(float(v))
        except Exception:
            pass
    return sorted(set(out))


def _candidate_exposures_from_dark_profile(
    con: sqlite3.Connection, dark_profile_id: int
) -> List[float]:
    if not _table_exists(con, "dark_library_exposures"):
        return []
    rows = con.execute(
        """
        SELECT DISTINCT exptime_s
        FROM dark_library_exposures
        WHERE dark_profile_id=?
          AND exptime_s IS NOT NULL
        """,
        (dark_profile_id,),
    ).fetchall()
    out: List[float] = []
    for (v,) in rows:
        try:
            out.append(float(v))
        except Exception:
            pass
    return sorted(set(out))


def _evaluate_exposure(
    con: sqlite3.Connection,
    sid: int,
    exptime_s: float,
    required_filters: Sequence[str],
    min_linear_headroom: float,
) -> Tuple[Optional[Dict[str, float]], int, List[str], Optional[float]]:
    """
    Returns:
      - per_filter_E_avg (or None if excluded)
      - n_used
      - reasons (machine-readable)
      - min_linear_headroom_seen (min across frames that were td.usable=1)
    """
    reasons: List[str] = []

    has_lh = _has_column(con, "training_derived_metrics", "linear_headroom_p99")

    cols = [
        "td.filter",
        "td.nebula_minus_bg_adu_s",
        "td.sky_ff_median_adu_s",
    ]
    if has_lh:
        cols.append("td.linear_headroom_p99")

    rows = con.execute(
        f"""
        SELECT {", ".join(cols)}
        FROM training_session_frames tsf
        JOIN training_derived_metrics td ON td.image_id = tsf.image_id
        WHERE tsf.training_session_id = ?
          AND td.usable = 1
          AND td.exptime_s = ?
          AND td.filter IS NOT NULL
          AND td.nebula_minus_bg_adu_s IS NOT NULL
          AND td.sky_ff_median_adu_s IS NOT NULL
        """,
        (sid, exptime_s),
    ).fetchall()

    per_filter_vals: Dict[str, List[float]] = {f.upper(): [] for f in required_filters}
    n_used = 0
    n_headroom_reject = 0
    min_lh_seen: Optional[float] = None

    for row in rows:
        f = str(row[0]).strip().upper()
        if f not in per_filter_vals:
            continue

        neb = _safe_float(row[1])
        sky = _safe_float(row[2])
        if neb is None or sky is None:
            continue

        if has_lh:
            lh = _safe_float(row[3])
            if lh is not None:
                min_lh_seen = (
                    lh if (min_lh_seen is None or lh < min_lh_seen) else min_lh_seen
                )
            if lh is not None and lh < min_linear_headroom:
                n_headroom_reject += 1
                continue

        E = _compute_E(neb, sky)
        if E is None:
            continue

        per_filter_vals[f].append(E)
        n_used += 1

    missing = [f for f in required_filters if len(per_filter_vals[f.upper()]) == 0]
    if missing:
        reasons.append(f"missing_filters:{','.join([m.upper() for m in missing])}")

    if n_used == 0:
        reasons.append("no_usable_frames_at_exposure")

    if n_headroom_reject > 0:
        reasons.append(f"rejected_by_linear_headroom:{n_headroom_reject}")

    if reasons:
        return None, n_used, reasons, min_lh_seen

    per_filter_E_avg: Dict[str, float] = {}
    for f in required_filters:
        vals = per_filter_vals[f.upper()]
        per_filter_E_avg[f.upper()] = sum(vals) / float(len(vals))

    return per_filter_E_avg, n_used, [], min_lh_seen


def _score_exposure(
    exptime_s: float,
    per_filter_E: Dict[str, float],
    min_exp: float,
    max_exp: float,
    w_mean: float,
    w_min: float,
    short_penalty: float,
    long_penalty: float,
) -> Tuple[float, float, float, float, float]:
    Es = list(per_filter_E.values())
    E_mean = sum(Es) / float(len(Es))
    E_min = min(Es)

    sp = short_penalty * (min_exp / exptime_s) if exptime_s > 0 else 1e9
    lp = long_penalty * (exptime_s / max_exp) if max_exp > 0 else 0.0

    score = (w_mean * E_mean) + (w_min * E_min) - sp - lp
    return score, E_mean, E_min, sp, lp


def _equal_snr_time_fractions(per_filter_E: Dict[str, float]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for f, E in per_filter_E.items():
        denom = E * E
        weights[f] = (1.0 / denom) if denom > 0 else 0.0
    s = sum(weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {f: 1.0 / n for f in weights}
    return {f: w / s for f, w in weights.items()}


def _ratio_vs_ha_from_time_fractions(
    required_filters: Sequence[str],
    time_fractions: Dict[str, float],
) -> Dict[str, float]:
    ref = (
        "HA"
        if "HA" in time_fractions
        else (list(time_fractions.keys())[0] if time_fractions else "HA")
    )
    ref_frac = time_fractions.get(ref, 1.0) or 1.0
    out: Dict[str, float] = {}
    for f in required_filters:
        fu = str(f).upper()
        out[fu] = (time_fractions.get(fu, 0.0) / ref_frac) if ref_frac > 0 else 0.0
    return out


def _aggregate_per_filter_E(
    evals: List[ExposureEval],
    required_filters: Sequence[str],
    *,
    weight_mode: str = "score_weighted",
) -> Optional[Dict[str, float]]:
    """
    Aggregate per-filter E across exposures to stabilize ratios when you have
    1 frame per filter per exposure.

    weight_mode:
      - "score_weighted": weight by shifted score (always positive)
      - "equal": equal weight
    """
    if not evals:
        return None

    # Only use evals that have all required filters (by construction they do)
    # Ensure stable weights even if scores are negative.
    scores = [float(e.score) for e in evals]
    min_s = min(scores)
    # shift so smallest score still gets a small positive weight
    eps = 1e-6

    def w_for(e: ExposureEval) -> float:
        if weight_mode == "equal":
            return 1.0
        # score_weighted (default)
        return max(float(e.score) - min_s + eps, eps)

    numer: Dict[str, float] = {str(f).upper(): 0.0 for f in required_filters}
    denom: float = 0.0

    for e in evals:
        w = w_for(e)
        denom += w
        for f in required_filters:
            fu = str(f).upper()
            numer[fu] += w * float(e.per_filter_E.get(fu, 0.0))

    if denom <= 0:
        return None

    return {fu: numer[fu] / denom for fu in numer.keys()}


def _plain_reason_not_chosen(
    *,
    excluded_reasons: Optional[List[str]],
    score: Optional[float],
    best_score: float,
    close_call_delta: float,
) -> str:
    if excluded_reasons:
        for r in excluded_reasons:
            if r.startswith("missing_filters:"):
                missing = r.split(":", 1)[1]
                return f"Not eligible: missing required filter(s): {missing}"
        for r in excluded_reasons:
            if r.startswith("rejected_by_linear_headroom:"):
                n = r.split(":", 1)[1]
                return f"Not eligible: {n} frame(s) failed linear headroom threshold"
        if "no_usable_frames_at_exposure" in excluded_reasons:
            return "Not eligible: no usable frames at this exposure"
        return "Not eligible: " + "; ".join(excluded_reasons)

    if score is None:
        return "Not chosen: no score (unexpected)"

    delta = best_score - score
    if delta <= 0:
        return "Chosen"
    if delta < close_call_delta:
        return f"Not chosen: close call, but slightly lower score than best (Δ={delta:.3f})"
    return f"Not chosen: lower overall score than best (Δ={delta:.3f})"


def write_training_result(
    con: sqlite3.Connection,
    sid: int,
    *,
    status: str,
    message: str,
    recommended_exptime_s: Optional[float] = None,
    recommended_filters: Optional[Sequence[str]] = None,
    recommended_time_fractions: Optional[Dict[str, float]] = None,
    recommended_ratio_vs_ha: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, object]] = None,
    conditions: Optional[Dict[str, object]] = None,
    stats: Optional[Dict[str, object]] = None,
) -> None:
    now = utc_now()
    con.execute(
        """
        INSERT INTO training_session_results (
            training_session_id,
            computed_utc,
            status,
            message,
            recommended_exptime_s,
            recommended_filters_json,
            recommended_time_fraction_json,
            recommended_ratio_vs_ha_json,
            constraints_json,
            conditions_json,
            stats_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(sid),
            now,
            status,
            message,
            recommended_exptime_s,
            (
                json.dumps(list(recommended_filters))
                if recommended_filters is not None
                else None
            ),
            (
                json.dumps(recommended_time_fractions)
                if recommended_time_fractions is not None
                else None
            ),
            (
                json.dumps(recommended_ratio_vs_ha)
                if recommended_ratio_vs_ha is not None
                else None
            ),
            json.dumps(constraints or {}),
            json.dumps(conditions or {}),
            json.dumps(stats or {}),
        ),
    )
    con.execute(
        "UPDATE training_sessions SET results_json=NULL, db_updated_utc=? WHERE training_session_id=?",
        (now, int(sid)),
    )


def solve_training_session(
    con: sqlite3.Connection,
    sid: int,
    *,
    # Global defaults passed in by caller (config), but can be overridden by session params_json.
    required_filters_default: Sequence[str],
    min_linear_headroom_default: float,
    w_mean_default: float,
    w_min_default: float,
    short_penalty_default: float,
    long_penalty_default: float,
    include_excluded_default: bool = True,
    close_call_delta_default: float = 0.05,
    # Ratio recommendation strategy defaults (config), can be overridden by session params_json.
    ratio_strategy_default: str = "aggregate_weighted",  # best | second | aggregate_weighted | aggregate_equal
    ratio_weight_mode_default: str = "score_weighted",  # score_weighted | equal
) -> Dict[str, object]:
    """
    Computes recommendation and writes a training_session_results row.
    Returns a dict for printing/monitor use.
    """
    if not _table_exists(con, "training_session_frames"):
        raise RuntimeError("training_session_frames table missing")
    if not _table_exists(con, "training_session_results"):
        raise RuntimeError("training_session_results table missing")
    if not _table_exists(con, "training_derived_metrics"):
        raise RuntimeError("training_derived_metrics table missing")

    sess = fetch_training_session(con, sid)
    params = parse_params_json(sess)

    # Session overrides (Option C behavior)
    required_filters = params.get("required_filters") or list(required_filters_default)
    required_filters = [
        str(f).strip().upper() for f in required_filters if str(f).strip()
    ]
    if not required_filters:
        required_filters = list(required_filters_default)

    min_linear_headroom = _safe_float(params.get("min_linear_headroom_p99"))
    if min_linear_headroom is None:
        min_linear_headroom = float(min_linear_headroom_default)

    w_mean = _safe_float(params.get("w_mean"))
    if w_mean is None:
        w_mean = float(w_mean_default)

    w_min = _safe_float(params.get("w_min"))
    if w_min is None:
        w_min = float(w_min_default)

    short_penalty = _safe_float(params.get("short_penalty"))
    if short_penalty is None:
        short_penalty = float(short_penalty_default)

    long_penalty = _safe_float(params.get("long_penalty"))
    if long_penalty is None:
        long_penalty = float(long_penalty_default)

    include_excluded = params.get("include_excluded")
    if include_excluded is None:
        include_excluded = bool(include_excluded_default)
    else:
        include_excluded = bool(include_excluded)

    close_call_delta = _safe_float(params.get("close_call_delta"))
    if close_call_delta is None:
        close_call_delta = float(close_call_delta_default)

    ratio_strategy = (
        str(params.get("ratio_strategy") or ratio_strategy_default).strip().lower()
    )
    ratio_weight_mode = (
        str(params.get("ratio_weight_mode") or ratio_weight_mode_default)
        .strip()
        .lower()
    )
    if ratio_strategy not in {
        "best",
        "second",
        "aggregate_weighted",
        "aggregate_equal",
    }:
        ratio_strategy = "aggregate_weighted"
    if ratio_weight_mode not in {"score_weighted", "equal"}:
        ratio_weight_mode = "score_weighted"

    constraints: Dict[str, object] = {
        "required_filters": list(required_filters),
        "min_linear_headroom_p99": float(min_linear_headroom),
        "weights": {
            "w_mean": float(w_mean),
            "w_min": float(w_min),
            "short_penalty": float(short_penalty),
            "long_penalty": float(long_penalty),
        },
        "ratio_strategy": ratio_strategy,
        "ratio_weight_mode": ratio_weight_mode,
    }

    # Candidate exposures: dark profile list first (if available), else session frames
    candidates: List[float] = []
    dark_profile_id = _infer_dark_profile_id(con, sid)
    if dark_profile_id is not None:
        dark_candidates = _candidate_exposures_from_dark_profile(con, dark_profile_id)
        if dark_candidates:
            candidates = dark_candidates
            constraints["candidate_source"] = "dark_library_exposures"
            constraints["dark_profile_id"] = dark_profile_id
        else:
            constraints["candidate_source"] = (
                "training_session_frames (dark profile had none)"
            )
    if not candidates:
        candidates = _candidate_exposures_for_session(con, sid)
        constraints["candidate_source"] = "training_session_frames"

    if not candidates:
        msg = "No candidate exposures found (no tagged frames with exptime_s)."
        write_training_result(
            con, sid, status="failed", message=msg, constraints=constraints, stats={}
        )
        return {
            "ok": False,
            "message": msg,
            "excluded": {},
            "best": None,
            "session": dict(sess),
        }

    min_exp = min(candidates)
    max_exp = max(candidates)

    evals: List[ExposureEval] = []
    excluded: Dict[str, List[str]] = {}
    exposure_report: Dict[str, Dict[str, object]] = {}

    for exp in candidates:
        per_filter_E, n_used, reasons, min_lh_seen = _evaluate_exposure(
            con,
            sid=sid,
            exptime_s=exp,
            required_filters=required_filters,
            min_linear_headroom=float(min_linear_headroom),
        )
        exp_key = f"{exp:.0f}"

        if per_filter_E is None:
            excluded[exp_key] = reasons or ["excluded"]
            exposure_report[exp_key] = {
                "filters": list(required_filters),
                "min_linear_headroom_p99": (
                    float(min_lh_seen) if min_lh_seen is not None else None
                ),
                "E_mean": None,
                "E_min": None,
                "score": None,
                "frames_used": int(n_used),
                "chosen": False,
                "reason": _plain_reason_not_chosen(
                    excluded_reasons=reasons,
                    score=None,
                    best_score=0.0,
                    close_call_delta=float(close_call_delta),
                ),
                "excluded_reasons": reasons,
            }
            continue

        score, E_mean, E_min, sp, lp = _score_exposure(
            exp,
            per_filter_E,
            min_exp=min_exp,
            max_exp=max_exp,
            w_mean=float(w_mean),
            w_min=float(w_min),
            short_penalty=float(short_penalty),
            long_penalty=float(long_penalty),
        )

        evals.append(
            ExposureEval(
                exptime_s=exp,
                per_filter_E=per_filter_E,
                E_mean=E_mean,
                E_min=E_min,
                score=score,
                n_frames_used=n_used,
                min_linear_headroom_p99=min_lh_seen,
            )
        )

        exposure_report[exp_key] = {
            "filters": list(required_filters),
            "min_linear_headroom_p99": (
                float(min_lh_seen) if min_lh_seen is not None else None
            ),
            "E_mean": float(E_mean),
            "E_min": float(E_min),
            "score": float(score),
            "frames_used": int(n_used),
            "chosen": False,
            "reason": "",
            "penalties": {"short": float(sp), "long": float(lp)},
            "E_per_filter": {k: float(v) for k, v in per_filter_E.items()},
        }

    if not evals:
        msg = "No exposures had usable data for all required filters."
        stats = {"excluded_candidates": excluded, "exposure_report": exposure_report}
        write_training_result(
            con, sid, status="failed", message=msg, constraints=constraints, stats=stats
        )
        return {
            "ok": False,
            "message": msg,
            "excluded": excluded,
            "best": None,
            "session": dict(sess),
        }

    evals.sort(key=lambda e: e.score, reverse=True)
    best = evals[0]
    second: Optional[ExposureEval] = evals[1] if len(evals) >= 2 else None

    best_key = f"{best.exptime_s:.0f}"

    for exp_key, rep in exposure_report.items():
        if exp_key == best_key:
            rep["chosen"] = True
            rep["reason"] = "Chosen: highest overall score"
        else:
            if rep.get("score") is not None:
                rep["chosen"] = False
                rep["reason"] = _plain_reason_not_chosen(
                    excluded_reasons=None,
                    score=float(rep["score"]),
                    best_score=float(best.score),
                    close_call_delta=float(close_call_delta),
                )

    # ---------------------------
    # RATIO COMPARISON
    # ---------------------------
    best_tf = _equal_snr_time_fractions(best.per_filter_E)
    best_ratio = _ratio_vs_ha_from_time_fractions(required_filters, best_tf)

    second_tf: Optional[Dict[str, float]] = None
    second_ratio: Optional[Dict[str, float]] = None
    second_delta_score: Optional[float] = None
    if second is not None:
        second_tf = _equal_snr_time_fractions(second.per_filter_E)
        second_ratio = _ratio_vs_ha_from_time_fractions(required_filters, second_tf)
        second_delta_score = float(best.score - second.score)

    agg_weight_mode = (
        "equal" if ratio_strategy == "aggregate_equal" else ratio_weight_mode
    )
    agg_per_filter_E = _aggregate_per_filter_E(
        evals, required_filters, weight_mode=agg_weight_mode
    )
    agg_tf: Optional[Dict[str, float]] = None
    agg_ratio: Optional[Dict[str, float]] = None
    if agg_per_filter_E is not None:
        agg_tf = _equal_snr_time_fractions(agg_per_filter_E)
        agg_ratio = _ratio_vs_ha_from_time_fractions(required_filters, agg_tf)

    # Choose the ratio that will be *recommended*
    used_ratio_source = ratio_strategy
    used_tf: Dict[str, float]
    used_ratio: Dict[str, float]
    if ratio_strategy == "best":
        used_tf = best_tf
        used_ratio = best_ratio
    elif (
        ratio_strategy == "second"
        and second_tf is not None
        and second_ratio is not None
    ):
        used_tf = second_tf
        used_ratio = second_ratio
    else:
        # aggregate_* (preferred)
        if agg_tf is not None and agg_ratio is not None:
            used_tf = agg_tf
            used_ratio = agg_ratio
        else:
            # fallback safety
            used_ratio_source = "best(fallback)"
            used_tf = best_tf
            used_ratio = best_ratio

    # ---------------------------
    # Write result + return
    # ---------------------------
    stats: Dict[str, object] = {
        "required_filters": list(required_filters),
        "recommended_exptime_s": float(best.exptime_s),
        "score": float(best.score),
        "E_mean": float(best.E_mean),
        "E_min": float(best.E_min),
        "frames_used": int(best.n_frames_used),
        "best_E_per_filter": {k: float(v) for k, v in best.per_filter_E.items()},
        "min_linear_headroom_p99": float(min_linear_headroom),
        "candidate_scores": [
            {
                "exptime_s": float(e.exptime_s),
                "score": float(e.score),
                "E_mean": float(e.E_mean),
                "E_min": float(e.E_min),
                "frames": int(e.n_frames_used),
                "min_linear_headroom_p99": (
                    float(e.min_linear_headroom_p99)
                    if e.min_linear_headroom_p99 is not None
                    else None
                ),
            }
            for e in evals
        ],
        "exposure_report": exposure_report,
        "ratio_comparison": {
            "used_ratio_source": used_ratio_source,
            "best": {
                "exptime_s": float(best.exptime_s),
                "ratio_vs_ha": best_ratio,
                "time_fractions": best_tf,
                "E_mean": float(best.E_mean),
                "E_min": float(best.E_min),
                "headroom_min": (
                    float(best.min_linear_headroom_p99)
                    if best.min_linear_headroom_p99 is not None
                    else None
                ),
                "score": float(best.score),
            },
            "second": (
                {
                    "exptime_s": float(second.exptime_s),
                    "ratio_vs_ha": second_ratio,
                    "time_fractions": second_tf,
                    "E_mean": float(second.E_mean),
                    "E_min": float(second.E_min),
                    "headroom_min": (
                        float(second.min_linear_headroom_p99)
                        if second.min_linear_headroom_p99 is not None
                        else None
                    ),
                    "score": float(second.score),
                    "delta_score": (
                        float(second_delta_score)
                        if second_delta_score is not None
                        else None
                    ),
                }
                if second is not None
                and second_ratio is not None
                and second_tf is not None
                else None
            ),
            "aggregate": (
                {
                    "ratio_vs_ha": agg_ratio,
                    "time_fractions": agg_tf,
                    "E_per_filter": agg_per_filter_E,
                    "weight_mode": agg_weight_mode,
                    "n_exposures": int(len(evals)),
                    "exposures": [float(e.exptime_s) for e in evals],
                    # headroom_min across eligible exps (use min of mins)
                    "headroom_min": (
                        float(
                            min(
                                [
                                    e.min_linear_headroom_p99
                                    for e in evals
                                    if e.min_linear_headroom_p99 is not None
                                ]
                            )
                        )
                        if any(e.min_linear_headroom_p99 is not None for e in evals)
                        else None
                    ),
                    "E_mean": float(sum([e.E_mean for e in evals]) / len(evals)),
                    "E_min": float(min([e.E_min for e in evals])),
                }
                if agg_ratio is not None
                and agg_tf is not None
                and agg_per_filter_E is not None
                else None
            ),
        },
    }
    if include_excluded:
        stats["excluded_candidates"] = excluded

    write_training_result(
        con,
        sid,
        status="ok",
        message="ok",
        recommended_exptime_s=best.exptime_s,
        recommended_filters=required_filters,
        recommended_time_fractions=used_tf,
        recommended_ratio_vs_ha=used_ratio,
        constraints=constraints,
        conditions={},
        stats=stats,
    )

    return {
        "ok": True,
        "best": {
            "recommended_exptime_s": float(best.exptime_s),
            "time_fractions": used_tf,
            "ratio_vs_ha": used_ratio,
            "score": float(best.score),
            "E_mean": float(best.E_mean),
            "E_min": float(best.E_min),
            "frames_used": int(best.n_frames_used),
            "E_per_filter": {k: float(v) for k, v in best.per_filter_E.items()},
            "ratio_used_source": used_ratio_source,
        },
        "excluded": excluded,
        "exposure_report": exposure_report,
        "candidates_scored": [float(e.exptime_s) for e in evals],
        "ratio_comparison": stats["ratio_comparison"],
        "session": {
            "training_session_id": int(sess["training_session_id"]),
            "target_name": sess["target_name"] or "",
            "watch_root_id": (
                sess["watch_root_id"] if sess["watch_root_id"] is not None else ""
            ),
            "status": sess["status"],
        },
    }


def close_training_session(con: sqlite3.Connection, sid: int) -> None:
    now = utc_now()
    con.execute(
        """
        UPDATE training_sessions
        SET status='closed', ended_utc=?, db_updated_utc=?
        WHERE training_session_id=?
        """,
        (now, now, int(sid)),
    )
