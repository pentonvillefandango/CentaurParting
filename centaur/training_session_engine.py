# centaur/training_session_engine.py
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Any

from centaur.logging import utc_now

DEFAULT_FILTERS = ["SII", "HA", "OIII"]


@dataclass(frozen=True)
class ExposureEval:
    exptime_s: float

    # Used for exposure scoring (STRICT: respects headroom threshold)
    per_filter_E: Dict[str, float]
    E_mean: float
    E_min: float
    score: float
    n_frames_used: int
    min_linear_headroom_p99: Optional[float]

    # Used for ratio estimation (LOOSE: may include headroom-rejected frames)
    per_filter_E_ratio: Dict[str, float]
    n_frames_ratio_used: int

    # Optional context (may be None if missing in table)
    avg_sky_limited_ratio: Optional[float]
    avg_transparency_proxy: Optional[float]  # sky_ff_median_adu_s proxy
    avg_nebula_over_sky: Optional[float]


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
    # E proxy used for ratios/scoring: signal rate divided by sqrt(sky rate)
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


def _quantile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    if q <= 0:
        return min(vals)
    if q >= 1:
        return max(vals)
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def _mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(sum(vals) / float(len(vals)))


def _stdev(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    m = _mean(vals)
    if m is None:
        return None
    v = sum((x - m) ** 2 for x in vals) / float(len(vals) - 1)
    return float(math.sqrt(v))


# -----------------------------
# Stage 4–5 add-ons: weighted robust aggregation helpers
# -----------------------------
def _weighted_quantile(
    values: List[float], weights: List[float], q: float
) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    pairs = []
    for v, w in zip(values, weights):
        try:
            vv = float(v)
            ww = float(w)
        except Exception:
            continue
        if not (math.isfinite(vv) and math.isfinite(ww)) or ww <= 0:
            continue
        pairs.append((vv, ww))
    if not pairs:
        return None
    pairs.sort(key=lambda x: x[0])
    total = sum(w for _v, w in pairs)
    if total <= 0:
        return None
    if q <= 0:
        return pairs[0][0]
    if q >= 1:
        return pairs[-1][0]
    target = q * total
    c = 0.0
    for v, w in pairs:
        c += w
        if c >= target:
            return v
    return pairs[-1][0]


def _weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    if not values or not weights or len(values) != len(weights):
        return None
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        try:
            vv = float(v)
            ww = float(w)
        except Exception:
            continue
        if not (math.isfinite(vv) and math.isfinite(ww)) or ww <= 0:
            continue
        num += vv * ww
        den += ww
    if den <= 0:
        return None
    return num / den


def _weighted_winsorized_mean(
    values: List[float], weights: List[float], *, lo_q: float, hi_q: float
) -> Optional[float]:
    lo = _weighted_quantile(values, weights, lo_q)
    hi = _weighted_quantile(values, weights, hi_q)
    if lo is None or hi is None:
        return _weighted_mean(values, weights)
    clipped: List[float] = []
    w2: List[float] = []
    for v, w in zip(values, weights):
        try:
            vv = float(v)
            ww = float(w)
        except Exception:
            continue
        if not (math.isfinite(vv) and math.isfinite(ww)) or ww <= 0:
            continue
        vv = min(max(vv, float(lo)), float(hi))
        clipped.append(vv)
        w2.append(ww)
    return _weighted_mean(clipped, w2)


def _evaluate_exposure(
    con: sqlite3.Connection,
    sid: int,
    exptime_s: float,
    required_filters: Sequence[str],
    min_linear_headroom: float,
    *,
    # Stage 4: allow ratios to include headroom-rejected frames
    ratio_include_headroom_rejected: bool = True,
) -> Tuple[
    Optional[Dict[str, float]],  # per_filter_E_avg (SCORING)
    Optional[Dict[str, float]],  # per_filter_E_avg_ratio (RATIO)
    int,  # n_used scoring
    int,  # n_used ratio
    List[str],  # reasons
    Optional[float],  # min_linear_headroom_seen among usable frames (any headroom)
    Optional[float],  # avg_sky_limited_ratio (scoring-used frames only)
    Optional[float],  # avg_transparency_proxy (scoring-used frames only)
    Optional[float],  # avg_nebula_over_sky (scoring-used frames only)
    int,  # n_headroom_reject
]:
    reasons: List[str] = []

    # optional columns
    has_lh = _has_column(con, "training_derived_metrics", "linear_headroom_p99")
    has_slr = _has_column(con, "training_derived_metrics", "sky_limited_ratio")
    has_tp = _has_column(con, "training_derived_metrics", "transparency_proxy")
    has_nos = _has_column(con, "training_derived_metrics", "nebula_over_sky")

    cols = ["td.filter", "td.nebula_minus_bg_adu_s", "td.sky_ff_median_adu_s"]
    if has_lh:
        cols.append("td.linear_headroom_p99")
    if has_slr:
        cols.append("td.sky_limited_ratio")
    if has_tp:
        cols.append("td.transparency_proxy")
    if has_nos:
        cols.append("td.nebula_over_sky")

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

    per_filter_vals_scoring: Dict[str, List[float]] = {
        f.upper(): [] for f in required_filters
    }
    per_filter_vals_ratio: Dict[str, List[float]] = {
        f.upper(): [] for f in required_filters
    }

    n_used_scoring = 0
    n_used_ratio = 0
    n_headroom_reject = 0

    min_lh_seen: Optional[float] = None

    slr_vals: List[float] = []
    tp_vals: List[float] = []
    nos_vals: List[float] = []

    idx = 3
    idx_lh = idx_slr = idx_tp = idx_nos = None
    if has_lh:
        idx_lh = idx
        idx += 1
    if has_slr:
        idx_slr = idx
        idx += 1
    if has_tp:
        idx_tp = idx
        idx += 1
    if has_nos:
        idx_nos = idx
        idx += 1

    for row in rows:
        f = str(row[0]).strip().upper()
        if f not in per_filter_vals_scoring:
            continue

        neb = _safe_float(row[1])
        sky = _safe_float(row[2])
        if neb is None or sky is None:
            continue

        lh: Optional[float] = None
        if idx_lh is not None:
            lh = _safe_float(row[idx_lh])
            if lh is not None:
                min_lh_seen = (
                    lh if (min_lh_seen is None or lh < min_lh_seen) else min_lh_seen
                )

        E = _compute_E(neb, sky)
        if E is None:
            continue

        # Ratio path: may include headroom-rejected frames
        if ratio_include_headroom_rejected or lh is None or lh >= min_linear_headroom:
            per_filter_vals_ratio[f].append(E)
            n_used_ratio += 1

        # Scoring path: respects headroom
        if lh is not None and lh < min_linear_headroom:
            n_headroom_reject += 1
            continue

        per_filter_vals_scoring[f].append(E)
        n_used_scoring += 1

        if idx_slr is not None:
            v = _safe_float(row[idx_slr])
            if v is not None:
                slr_vals.append(v)
        if idx_tp is not None:
            v = _safe_float(row[idx_tp])
            if v is not None:
                tp_vals.append(v)
        if idx_nos is not None:
            v = _safe_float(row[idx_nos])
            if v is not None:
                nos_vals.append(v)

    missing_scoring = [
        f for f in required_filters if len(per_filter_vals_scoring[f.upper()]) == 0
    ]
    if missing_scoring:
        reasons.append(
            f"missing_filters:{','.join([m.upper() for m in missing_scoring])}"
        )

    if n_used_scoring == 0:
        reasons.append("no_usable_frames_at_exposure")

    if n_headroom_reject > 0:
        reasons.append(f"rejected_by_linear_headroom:{n_headroom_reject}")

    per_filter_E_avg_ratio: Optional[Dict[str, float]] = None
    missing_ratio = [
        f for f in required_filters if len(per_filter_vals_ratio[f.upper()]) == 0
    ]
    if not missing_ratio:
        per_filter_E_avg_ratio = {
            f.upper(): float(
                sum(per_filter_vals_ratio[f.upper()])
                / float(len(per_filter_vals_ratio[f.upper()]))
            )
            for f in required_filters
        }

    if reasons:
        return (
            None,
            per_filter_E_avg_ratio,
            n_used_scoring,
            n_used_ratio,
            reasons,
            min_lh_seen,
            _mean(slr_vals),
            _mean(tp_vals),
            _mean(nos_vals),
            n_headroom_reject,
        )

    per_filter_E_avg_scoring: Dict[str, float] = {
        f.upper(): float(
            sum(per_filter_vals_scoring[f.upper()])
            / float(len(per_filter_vals_scoring[f.upper()]))
        )
        for f in required_filters
    }

    return (
        per_filter_E_avg_scoring,
        per_filter_E_avg_ratio,
        n_used_scoring,
        n_used_ratio,
        [],
        min_lh_seen,
        _mean(slr_vals),
        _mean(tp_vals),
        _mean(nos_vals),
        n_headroom_reject,
    )


def _score_exposure(
    exptime_s: float,
    per_filter_E: Dict[str, float],
    min_exp: float,
    max_exp: float,
    w_mean: float,
    w_min: float,
    short_penalty: float,
    long_penalty: float,
    *,
    avg_sky_limited_ratio: Optional[float],
    sky_limited_target_ratio: float,
    sky_limited_penalty_weight: float,
) -> Tuple[float, float, float, float, float, float]:
    Es = list(per_filter_E.values())
    E_mean = sum(Es) / float(len(Es))
    E_min = min(Es)

    sp = short_penalty * (min_exp / exptime_s) if exptime_s > 0 else 1e9
    lp = long_penalty * (exptime_s / max_exp) if max_exp > 0 else 0.0

    sky_pen = 0.0
    if (
        sky_limited_penalty_weight is not None
        and sky_limited_penalty_weight > 0
        and avg_sky_limited_ratio is not None
        and sky_limited_target_ratio is not None
        and sky_limited_target_ratio > 0
    ):
        r = float(avg_sky_limited_ratio)
        tgt = float(sky_limited_target_ratio)
        if r < tgt:
            frac = max(0.0, min(1.0, (tgt - r) / tgt))
            sky_pen = float(sky_limited_penalty_weight) * frac

    score = (w_mean * E_mean) + (w_min * E_min) - sp - lp - sky_pen
    return score, E_mean, E_min, sp, lp, sky_pen


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
    required_filters_default: Sequence[str],
    min_linear_headroom_default: float,
    w_mean_default: float,
    w_min_default: float,
    short_penalty_default: float,
    long_penalty_default: float,
    include_excluded_default: bool = True,
    close_call_delta_default: float = 0.05,
    # Existing 1–3 defaults
    sky_limited_target_ratio_default: float = 10.0,
    sky_limited_penalty_weight_default: float = 0.08,
    transparency_variation_warn_frac_default: float = 0.35,
    prefer_aggregate_ratio_when_unstable_default: bool = True,
    aggregate_ratio_weight_mode_default: str = "nebula_over_sky",  # or "uniform"
    # Stage 4–5 defaults
    ratio_include_headroom_rejected_default: bool = True,
    ratio_min_samples_per_filter_default: int = 2,
    ratio_variation_warn_cv_default: float = 0.40,
    # NEW: robust aggregate controls (stages 4–5 enhancement)
    aggregate_ratio_method_default: str = "winsor",  # mean|median|winsor
    aggregate_ratio_winsor_q_default: float = 0.15,  # winsor clamp [q, 1-q]
    aggregate_transparency_weight_clamp_default: float = 4.0,  # clamp tp weight factor
) -> Dict[str, object]:
    if not _table_exists(con, "training_session_frames"):
        raise RuntimeError("training_session_frames table missing")
    if not _table_exists(con, "training_session_results"):
        raise RuntimeError("training_session_results table missing")
    if not _table_exists(con, "training_derived_metrics"):
        raise RuntimeError("training_derived_metrics table missing")

    sess = fetch_training_session(con, sid)
    params = parse_params_json(sess)

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

    sky_limited_target_ratio = _safe_float(params.get("sky_limited_target_ratio"))
    if sky_limited_target_ratio is None:
        sky_limited_target_ratio = float(sky_limited_target_ratio_default)

    sky_limited_penalty_weight = _safe_float(params.get("sky_limited_penalty_weight"))
    if sky_limited_penalty_weight is None:
        sky_limited_penalty_weight = float(sky_limited_penalty_weight_default)

    transparency_variation_warn_frac = _safe_float(
        params.get("transparency_variation_warn_frac")
    )
    if transparency_variation_warn_frac is None:
        transparency_variation_warn_frac = float(
            transparency_variation_warn_frac_default
        )

    prefer_aggregate_ratio_when_unstable = params.get(
        "prefer_aggregate_ratio_when_unstable"
    )
    if prefer_aggregate_ratio_when_unstable is None:
        prefer_aggregate_ratio_when_unstable = bool(
            prefer_aggregate_ratio_when_unstable_default
        )
    else:
        prefer_aggregate_ratio_when_unstable = bool(
            prefer_aggregate_ratio_when_unstable
        )

    aggregate_ratio_weight_mode = (
        str(
            params.get("aggregate_ratio_weight_mode")
            or aggregate_ratio_weight_mode_default
        )
        .strip()
        .lower()
    )
    # EDIT: allow more modes
    if aggregate_ratio_weight_mode not in (
        "nebula_over_sky",
        "uniform",
        "inverse_transparency",
        "combined",
    ):
        aggregate_ratio_weight_mode = "nebula_over_sky"

    # Stage 4–5 overrides
    ratio_include_headroom_rejected = params.get("ratio_include_headroom_rejected")
    if ratio_include_headroom_rejected is None:
        ratio_include_headroom_rejected = bool(ratio_include_headroom_rejected_default)
    else:
        ratio_include_headroom_rejected = bool(ratio_include_headroom_rejected)

    ratio_min_samples_per_filter = _safe_int(params.get("ratio_min_samples_per_filter"))
    if ratio_min_samples_per_filter is None:
        ratio_min_samples_per_filter = int(ratio_min_samples_per_filter_default)
    ratio_min_samples_per_filter = max(1, int(ratio_min_samples_per_filter))

    ratio_variation_warn_cv = _safe_float(params.get("ratio_variation_warn_cv"))
    if ratio_variation_warn_cv is None:
        ratio_variation_warn_cv = float(ratio_variation_warn_cv_default)

    # NEW: robust aggregate overrides
    aggregate_ratio_method = (
        str(params.get("aggregate_ratio_method") or aggregate_ratio_method_default)
        .strip()
        .lower()
    )
    if aggregate_ratio_method not in ("mean", "median", "winsor"):
        aggregate_ratio_method = "winsor"

    aggregate_ratio_winsor_q = _safe_float(params.get("aggregate_ratio_winsor_q"))
    if aggregate_ratio_winsor_q is None:
        aggregate_ratio_winsor_q = float(aggregate_ratio_winsor_q_default)
    aggregate_ratio_winsor_q = max(0.0, min(0.49, float(aggregate_ratio_winsor_q)))

    aggregate_transparency_weight_clamp = _safe_float(
        params.get("aggregate_transparency_weight_clamp")
    )
    if aggregate_transparency_weight_clamp is None:
        aggregate_transparency_weight_clamp = float(
            aggregate_transparency_weight_clamp_default
        )
    aggregate_transparency_weight_clamp = max(
        1.0, float(aggregate_transparency_weight_clamp)
    )

    constraints: Dict[str, object] = {
        "required_filters": list(required_filters),
        "min_linear_headroom_p99": float(min_linear_headroom),
        "weights": {
            "w_mean": float(w_mean),
            "w_min": float(w_min),
            "short_penalty": float(short_penalty),
            "long_penalty": float(long_penalty),
        },
        "sky_limited": {
            "target_ratio": float(sky_limited_target_ratio),
            "penalty_weight": float(sky_limited_penalty_weight),
        },
        "transparency": {
            "warn_frac": float(transparency_variation_warn_frac),
            "prefer_aggregate_ratio_when_unstable": bool(
                prefer_aggregate_ratio_when_unstable
            ),
        },
        # EDIT: keep old key for backward familiarity
        "aggregate_ratio_weight_mode": aggregate_ratio_weight_mode,
        # NEW: structured aggregate info
        "aggregate_ratio": {
            "weight_mode": aggregate_ratio_weight_mode,
            "method": aggregate_ratio_method,
            "winsor_q": float(aggregate_ratio_winsor_q),
            "transparency_weight_clamp": float(aggregate_transparency_weight_clamp),
        },
        "ratio": {
            "include_headroom_rejected": bool(ratio_include_headroom_rejected),
            "min_samples_per_filter": int(ratio_min_samples_per_filter),
            "variation_warn_cv": float(ratio_variation_warn_cv),
        },
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
    penalty_report: Dict[str, Dict[str, float]] = {}

    for exp in candidates:
        (
            per_filter_E_scoring,
            per_filter_E_ratio,
            n_used_scoring,
            n_used_ratio,
            reasons,
            min_lh_seen,
            avg_slr,
            avg_tp,
            avg_nos,
            n_headroom_reject,
        ) = _evaluate_exposure(
            con,
            sid=sid,
            exptime_s=exp,
            required_filters=required_filters,
            min_linear_headroom=float(min_linear_headroom),
            ratio_include_headroom_rejected=bool(ratio_include_headroom_rejected),
        )

        exp_key = f"{exp:.0f}"

        if per_filter_E_scoring is None:
            excluded[exp_key] = reasons or ["excluded"]
            exposure_report[exp_key] = {
                "filters": list(required_filters),
                "min_linear_headroom_p99": (
                    float(min_lh_seen) if min_lh_seen is not None else None
                ),
                "E_mean": None,
                "E_min": None,
                "score": None,
                "frames_used": int(n_used_scoring),
                "chosen": False,
                "reason": _plain_reason_not_chosen(
                    excluded_reasons=reasons,
                    score=None,
                    best_score=0.0,
                    close_call_delta=float(close_call_delta),
                ),
                "excluded_reasons": reasons,
                "avg_sky_limited_ratio": avg_slr,
                "avg_transparency_proxy": avg_tp,
                "avg_nebula_over_sky": avg_nos,
                # Stage 4 visibility
                "frames_ratio_used": int(n_used_ratio),
                "E_per_filter_ratio": (
                    {k: float(v) for k, v in (per_filter_E_ratio or {}).items()}
                    if per_filter_E_ratio is not None
                    else None
                ),
                "headroom_reject_count": int(n_headroom_reject),
            }
            continue

        score, E_mean, E_min, sp, lp, sky_pen = _score_exposure(
            exp,
            per_filter_E_scoring,
            min_exp=min_exp,
            max_exp=max_exp,
            w_mean=float(w_mean),
            w_min=float(w_min),
            short_penalty=float(short_penalty),
            long_penalty=float(long_penalty),
            avg_sky_limited_ratio=avg_slr,
            sky_limited_target_ratio=float(sky_limited_target_ratio),
            sky_limited_penalty_weight=float(sky_limited_penalty_weight),
        )

        per_filter_E_ratio_final = per_filter_E_ratio or dict(per_filter_E_scoring)

        evals.append(
            ExposureEval(
                exptime_s=exp,
                per_filter_E=per_filter_E_scoring,
                E_mean=E_mean,
                E_min=E_min,
                score=score,
                n_frames_used=n_used_scoring,
                min_linear_headroom_p99=min_lh_seen,
                per_filter_E_ratio=per_filter_E_ratio_final,
                n_frames_ratio_used=n_used_ratio,
                avg_sky_limited_ratio=avg_slr,
                avg_transparency_proxy=avg_tp,
                avg_nebula_over_sky=avg_nos,
            )
        )

        penalty_report[exp_key] = {
            "short": float(sp),
            "long": float(lp),
            "sky_limited": float(sky_pen),
        }

        exposure_report[exp_key] = {
            "filters": list(required_filters),
            "min_linear_headroom_p99": (
                float(min_lh_seen) if min_lh_seen is not None else None
            ),
            "E_mean": float(E_mean),
            "E_min": float(E_min),
            "score": float(score),
            "frames_used": int(n_used_scoring),
            "chosen": False,
            "reason": "",
            "penalties": {
                "short": float(sp),
                "long": float(lp),
                "sky_limited": float(sky_pen),
            },
            "E_per_filter": {k: float(v) for k, v in per_filter_E_scoring.items()},
            "avg_sky_limited_ratio": (float(avg_slr) if avg_slr is not None else None),
            "avg_transparency_proxy": (float(avg_tp) if avg_tp is not None else None),
            "avg_nebula_over_sky": (float(avg_nos) if avg_nos is not None else None),
            # Stage 4 visibility
            "frames_ratio_used": int(n_used_ratio),
            "E_per_filter_ratio": {
                k: float(v) for k, v in per_filter_E_ratio_final.items()
            },
            "headroom_reject_count": int(n_headroom_reject),
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
    second = evals[1] if len(evals) >= 2 else None

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

    # -----------------------------
    # Ratios (Stage 4): use per_filter_E_ratio
    # -----------------------------
    def ratios_from_per_filter_E(
        per_filter_E: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        tf = _equal_snr_time_fractions(per_filter_E)
        ref = "HA" if "HA" in tf else list(tf.keys())[0]
        ref_frac = tf.get(ref, 1.0) or 1.0
        r = {
            f.upper(): (tf.get(f.upper(), 0.0) / ref_frac if ref_frac > 0 else 0.0)
            for f in required_filters
        }
        return tf, r

    best_tf, best_ratio = ratios_from_per_filter_E(best.per_filter_E_ratio)

    second_tf: Optional[Dict[str, float]] = None
    second_ratio: Optional[Dict[str, float]] = None
    if second is not None:
        second_tf, second_ratio = ratios_from_per_filter_E(second.per_filter_E_ratio)

    # -----------------------------
    # EDIT: robust + transparency-aware aggregate ratios
    # -----------------------------
    tp_vals_all = [
        float(e.avg_transparency_proxy)
        for e in evals
        if e.avg_transparency_proxy is not None
        and math.isfinite(float(e.avg_transparency_proxy))
    ]
    tp_med = _quantile(tp_vals_all, 0.50) if tp_vals_all else None

    def _tp_weight(e: ExposureEval) -> float:
        if tp_med is None:
            return 1.0
        tp = e.avg_transparency_proxy
        if tp is None:
            return 1.0
        try:
            tpf = float(tp)
        except Exception:
            return 1.0
        if not math.isfinite(tpf) or tpf <= 0:
            return 1.0
        raw = float(tp_med) / float(tpf)  # darker-than-median => >1, brighter => <1
        clamp = float(aggregate_transparency_weight_clamp)
        return max(1.0 / clamp, min(clamp, raw))

    def _base_weight(e: ExposureEval) -> float:
        if aggregate_ratio_weight_mode == "uniform":
            return 1.0
        if aggregate_ratio_weight_mode == "nebula_over_sky":
            w = (
                float(e.avg_nebula_over_sky)
                if e.avg_nebula_over_sky is not None
                else 1.0
            )
            return w if (math.isfinite(w) and w > 0) else 1.0
        if aggregate_ratio_weight_mode == "inverse_transparency":
            return 1.0  # tp applied separately
        if aggregate_ratio_weight_mode == "combined":
            w = (
                float(e.avg_nebula_over_sky)
                if e.avg_nebula_over_sky is not None
                else 1.0
            )
            w = w if (math.isfinite(w) and w > 0) else 1.0
            return w
        return 1.0

    def _final_weight(e: ExposureEval) -> float:
        w = _base_weight(e)
        if aggregate_ratio_weight_mode in ("inverse_transparency", "combined"):
            w *= _tp_weight(e)
        if not math.isfinite(w) or w <= 0:
            return 1.0
        return float(w)

    agg_used_exps: List[int] = [int(round(float(e.exptime_s))) for e in evals]
    weights = [_final_weight(e) for e in evals]

    agg_per_filter_E: Dict[str, float] = {}
    for f in required_filters:
        vals: List[float] = []
        ws: List[float] = []
        for e, w in zip(evals, weights):
            v = e.per_filter_E_ratio.get(f.upper())
            if v is None:
                continue
            vf = float(v)
            if not math.isfinite(vf):
                continue
            vals.append(vf)
            ws.append(float(w))

        if not vals:
            agg_per_filter_E[f.upper()] = 0.0
            continue

        if aggregate_ratio_method == "mean":
            m = _weighted_mean(vals, ws)
        elif aggregate_ratio_method == "median":
            m = _weighted_quantile(vals, ws, 0.50)
        else:
            q = float(aggregate_ratio_winsor_q)
            m = _weighted_winsorized_mean(vals, ws, lo_q=q, hi_q=1.0 - q)

        if m is None or not math.isfinite(float(m)):
            m = _weighted_mean(vals, ws)
        if m is None or not math.isfinite(float(m)):
            m = float(sum(vals) / len(vals))

        agg_per_filter_E[f.upper()] = float(m)

    agg_tf, agg_ratio = ratios_from_per_filter_E(agg_per_filter_E)

    agg_Es = list(agg_per_filter_E.values()) or [0.0]
    agg_E_mean = sum(agg_Es) / float(len(agg_Es))
    agg_E_min = min(agg_Es)
    agg_head_min = min(
        [
            e.min_linear_headroom_p99
            for e in evals
            if e.min_linear_headroom_p99 is not None
        ],
        default=None,
    )

    # -----------------------------
    # Transparency variation detection (existing)
    # -----------------------------
    tp_vals = [
        e.avg_transparency_proxy for e in evals if e.avg_transparency_proxy is not None
    ]
    transparency_warning: Optional[str] = None
    transparency_stats: Dict[str, Any] = {}

    if len(tp_vals) >= 2:
        p10 = _quantile([float(x) for x in tp_vals], 0.10)
        p50 = _quantile([float(x) for x in tp_vals], 0.50)
        p90 = _quantile([float(x) for x in tp_vals], 0.90)
        if p10 is not None and p50 is not None and p90 is not None and p50 > 0:
            spread_frac = float(p90 - p10) / float(p50)
            transparency_stats = {
                "p10": float(p10),
                "p50": float(p50),
                "p90": float(p90),
                "spread_frac": float(spread_frac),
            }
            if spread_frac >= float(transparency_variation_warn_frac):
                transparency_warning = (
                    f"Sky conditions varied across exposures (transparency spread ~{spread_frac:.2f} of median). "
                    f"Ratios may be unstable; aggregate ratio recommended."
                )

    # -----------------------------
    # Stage 5: ratio stability + sample counts (existing)
    # -----------------------------
    ratio_samples_per_filter: Dict[str, List[float]] = {
        f.upper(): [] for f in required_filters
    }
    for e in evals:
        for f in required_filters:
            v = e.per_filter_E_ratio.get(f.upper())
            if v is not None and math.isfinite(float(v)):
                ratio_samples_per_filter[f.upper()].append(float(v))

    ratio_sample_counts = {k: len(v) for k, v in ratio_samples_per_filter.items()}

    ratio_variation_cv: Dict[str, Optional[float]] = {}
    for f, vals in ratio_samples_per_filter.items():
        m = _mean(vals)
        sd = _stdev(vals)
        if m is None or sd is None or m <= 0:
            ratio_variation_cv[f] = None
        else:
            ratio_variation_cv[f] = float(sd / m)

    ratio_warning_parts: List[str] = []
    for f in required_filters:
        k = f.upper()
        if ratio_sample_counts.get(k, 0) < ratio_min_samples_per_filter:
            ratio_warning_parts.append(
                f"{k} has only {ratio_sample_counts.get(k, 0)} sample(s)"
            )
        cv = ratio_variation_cv.get(k)
        if cv is not None and cv >= float(ratio_variation_warn_cv):
            ratio_warning_parts.append(f"{k} variation high (cv~{cv:.2f})")

    ratio_stability_warning = None
    if ratio_warning_parts:
        ratio_stability_warning = "Ratio stability warning: " + "; ".join(
            ratio_warning_parts
        )

    # Choose which ratio to *recommend*
    ratio_source = "best"
    recommended_tf = best_tf
    recommended_ratio = best_ratio

    if transparency_warning and prefer_aggregate_ratio_when_unstable:
        ratio_source = "aggregate"
        recommended_tf = agg_tf
        recommended_ratio = agg_ratio

    if ratio_stability_warning:
        ratio_source = "aggregate"
        recommended_tf = agg_tf
        recommended_ratio = agg_ratio

    # -----------------------------
    # Build stats
    # -----------------------------
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
                "frames_ratio": int(e.n_frames_ratio_used),
                "min_linear_headroom_p99": (
                    float(e.min_linear_headroom_p99)
                    if e.min_linear_headroom_p99 is not None
                    else None
                ),
                "avg_sky_limited_ratio": (
                    float(e.avg_sky_limited_ratio)
                    if e.avg_sky_limited_ratio is not None
                    else None
                ),
                "avg_transparency_proxy": (
                    float(e.avg_transparency_proxy)
                    if e.avg_transparency_proxy is not None
                    else None
                ),
                "avg_nebula_over_sky": (
                    float(e.avg_nebula_over_sky)
                    if e.avg_nebula_over_sky is not None
                    else None
                ),
            }
            for e in evals
        ],
        "exposure_report": exposure_report,
        "penalty_report": penalty_report,
        "ratios": {
            "source_used": ratio_source,
            "best": {
                "exptime_s": float(best.exptime_s),
                "time_fractions": best_tf,
                "ratio_vs_ha": best_ratio,
                "E_mean": float(best.E_mean),
                "E_min": float(best.E_min),
                "headroom_min": (
                    float(best.min_linear_headroom_p99)
                    if best.min_linear_headroom_p99 is not None
                    else None
                ),
                "score": float(best.score),
                "ratio_includes_headroom_rejected": bool(
                    ratio_include_headroom_rejected
                ),
            },
            "second": (
                {
                    "exptime_s": float(second.exptime_s),
                    "time_fractions": second_tf,
                    "ratio_vs_ha": second_ratio,
                    "E_mean": float(second.E_mean),
                    "E_min": float(second.E_min),
                    "headroom_min": (
                        float(second.min_linear_headroom_p99)
                        if second.min_linear_headroom_p99 is not None
                        else None
                    ),
                    "score": float(second.score),
                    "delta_score": float(best.score - second.score),
                    "ratio_includes_headroom_rejected": bool(
                        ratio_include_headroom_rejected
                    ),
                }
                if second is not None
                and second_tf is not None
                and second_ratio is not None
                else None
            ),
            "aggregate": {
                "exptime_s": None,
                "time_fractions": agg_tf,
                "ratio_vs_ha": agg_ratio,
                "E_mean": float(agg_E_mean),
                "E_min": float(agg_E_min),
                "headroom_min": (
                    float(agg_head_min) if agg_head_min is not None else None
                ),
                "exposures_used": sorted(set(agg_used_exps)),
                "weight_mode": aggregate_ratio_weight_mode,
                "method": aggregate_ratio_method,
                "winsor_q": float(aggregate_ratio_winsor_q),
                "tp_median": (float(tp_med) if tp_med is not None else None),
                "tp_weight_clamp": float(aggregate_transparency_weight_clamp),
                "ratio_includes_headroom_rejected": bool(
                    ratio_include_headroom_rejected
                ),
            },
            "warnings": {
                "transparency": transparency_warning,
                "transparency_stats": transparency_stats,
                "ratio_stability": ratio_stability_warning,
                "ratio_sample_counts": ratio_sample_counts,
                "ratio_variation_cv": ratio_variation_cv,
            },
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
        recommended_time_fractions=recommended_tf,
        recommended_ratio_vs_ha=recommended_ratio,
        constraints=constraints,
        conditions={},
        stats=stats,
    )

    return {
        "ok": True,
        "best": {
            "recommended_exptime_s": float(best.exptime_s),
            "time_fractions": recommended_tf,
            "ratio_vs_ha": recommended_ratio,
            "ratio_source": ratio_source,
            "score": float(best.score),
            "E_mean": float(best.E_mean),
            "E_min": float(best.E_min),
            "frames_used": int(best.n_frames_used),
            "E_per_filter": {k: float(v) for k, v in best.per_filter_E.items()},
        },
        "excluded": excluded,
        "exposure_report": exposure_report,
        "candidates_scored": [float(e.exptime_s) for e in evals],
        "ratios": stats["ratios"],
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
