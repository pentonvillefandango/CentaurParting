#!/usr/bin/env python3
# centaur/tools/training_session_close.py
from __future__ import annotations

import argparse
import sqlite3
from typing import Any, Dict, Tuple, List, Optional

from centaur.config import default_config
from centaur.training_session_engine import (
    solve_training_session,
    close_training_session,
)

# -----------------------------
# Small helpers
# -----------------------------


def _round(x: Any, nd: int = 3) -> Any:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return x
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return float(round(x, nd))
        return x
    except Exception:
        return x


def _round_recursive(obj: Any, nd: int = 3) -> Any:
    if isinstance(obj, dict):
        return {str(k): _round_recursive(v, nd=nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_recursive(v, nd=nd) for v in obj]
    return _round(obj, nd=nd)


def _sort_exposure_keys(keys: List[str]) -> List[str]:
    def key_fn(k: str) -> float:
        try:
            return float(k)
        except Exception:
            return 1e18

    return sorted(keys, key=key_fn)


def _fmt_filters(filters: Any) -> str:
    if not filters:
        return "-"
    try:
        s = [str(x).upper().strip() for x in list(filters)]
    except Exception:
        return "-"
    short = []
    for f in s:
        if f.startswith("H"):
            short.append("H")
        elif f.startswith("O"):
            short.append("O")
        elif f.startswith("S"):
            short.append("S")
        else:
            short.append(f[:1])
    return ",".join(short)


def _fmt_float(x: Any, nd: int = 3, width: int = 0) -> str:
    if x is None:
        s = "—"
    else:
        try:
            s = f"{float(x):.{nd}f}"
        except Exception:
            s = "—"
    return s.rjust(width) if width > 0 else s


def _fmt_score(x: Any) -> str:
    return _fmt_float(x, nd=3)


def _coerce_num(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _equal_snr_time_fractions(per_filter_E: Dict[str, float]) -> Dict[str, float]:
    """
    Same logic as engine:
      time ∝ 1/E^2  (equalize SNR)
    """
    weights: Dict[str, float] = {}
    for f, E in per_filter_E.items():
        try:
            denom = float(E) * float(E)
            weights[f] = (1.0 / denom) if denom > 0 else 0.0
        except Exception:
            weights[f] = 0.0

    s = sum(weights.values())
    if s <= 0:
        n = len(weights) or 1
        return {f: 1.0 / n for f in weights}
    return {f: w / s for f, w in weights.items()}


def _ratio_vs_ha_from_E(per_filter_E: Dict[str, float]) -> Dict[str, float]:
    tf = _equal_snr_time_fractions(per_filter_E)
    ref = "HA" if "HA" in tf else (list(tf.keys())[0] if tf else "HA")
    ref_frac = tf.get(ref, 1.0) or 1.0
    out: Dict[str, float] = {}
    for k, v in tf.items():
        out[str(k).upper()] = (float(v) / float(ref_frac)) if ref_frac > 0 else 0.0
    return out


def _stable_ratio_line(ratios: Dict[str, Any]) -> str:
    if not ratios:
        return ""
    parts: List[str] = []
    for k in ["SII", "HA", "OIII"]:
        if k in ratios:
            try:
                parts.append(f"{k} {float(ratios[k]):.2f}x")
            except Exception:
                parts.append(f"{k} —")
    if not parts:
        for k, v in ratios.items():
            try:
                parts.append(f"{k} {float(v):.2f}x")
            except Exception:
                parts.append(f"{k} —")
    return " | ".join(parts)


def _build_comparisons_from_exposure_report(
    exposure_report: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produces:
      - best (from chosen=True)
      - second_best (highest score among non-chosen, non-excluded)
      - aggregate (across all eligible scored exposures)
    """
    # Eligible scored candidates (score not None and not excluded)
    eligible: List[Tuple[str, Dict[str, Any], float]] = []
    chosen_key: Optional[str] = None

    for k, rep in exposure_report.items():
        score = _coerce_num(rep.get("score"))
        excluded = rep.get("excluded_reasons")
        if rep.get("chosen"):
            chosen_key = k
        if score is not None and not excluded:
            eligible.append((k, rep, float(score)))

    eligible.sort(key=lambda t: t[2], reverse=True)

    best: Optional[Dict[str, Any]] = None
    second: Optional[Dict[str, Any]] = None
    agg: Optional[Dict[str, Any]] = None

    def pack(
        k: str, rep: Dict[str, Any], score: float, *, label: str
    ) -> Dict[str, Any]:
        exp_s = None
        try:
            exp_s = int(round(float(k)))
        except Exception:
            exp_s = k

        Epf = rep.get("E_per_filter") or {}
        # Normalize keys
        Epf2: Dict[str, float] = {}
        for fk, fv in dict(Epf).items():
            vv = _coerce_num(fv)
            if vv is not None:
                Epf2[str(fk).upper()] = float(vv)

        ratios = _ratio_vs_ha_from_E(Epf2) if Epf2 else {}
        return {
            "label": label,
            "exptime_s": exp_s,
            "score": _coerce_num(score),
            "E_mean": _coerce_num(rep.get("E_mean")),
            "E_min": _coerce_num(rep.get("E_min")),
            "min_linear_headroom_p99": _coerce_num(rep.get("min_linear_headroom_p99")),
            "filters": rep.get("filters"),
            "ratio_vs_ha": ratios,
        }

    # Best: prefer chosen=True, else top-scoring eligible
    if chosen_key is not None:
        rep = exposure_report.get(chosen_key) or {}
        sc = _coerce_num(rep.get("score"))
        if sc is not None and not rep.get("excluded_reasons"):
            best = pack(chosen_key, rep, float(sc), label="BEST")
    if best is None and eligible:
        k, rep, sc = eligible[0]
        best = pack(k, rep, sc, label="BEST")

    # Second best: next eligible after best key
    if eligible:
        for k, rep, sc in eligible:
            if best is not None and str(k) == str(best.get("exptime_s")).replace(
                "s", ""
            ):
                # Not reliable; compare keys instead
                pass
        best_key = None
        if best is not None:
            # best["exptime_s"] may be int; we want original key
            # safest: use chosen_key if present, else eligible[0][0]
            best_key = chosen_key or eligible[0][0]

        for k, rep, sc in eligible:
            if best_key is not None and str(k) == str(best_key):
                continue
            second = pack(k, rep, sc, label="2ND")
            break

    # Aggregate: average E_per_filter across ALL eligible exposures
    if eligible:
        sum_E: Dict[str, float] = {}
        cnt_E: Dict[str, int] = {}
        exp_list: List[int] = []
        min_head: Optional[float] = None

        for k, rep, _sc in eligible:
            try:
                exp_list.append(int(round(float(k))))
            except Exception:
                pass

            lh = _coerce_num(rep.get("min_linear_headroom_p99"))
            if lh is not None:
                min_head = lh if (min_head is None or lh < min_head) else min_head

            Epf = rep.get("E_per_filter") or {}
            for fk, fv in dict(Epf).items():
                vv = _coerce_num(fv)
                if vv is None:
                    continue
                fk2 = str(fk).upper()
                sum_E[fk2] = sum_E.get(fk2, 0.0) + float(vv)
                cnt_E[fk2] = cnt_E.get(fk2, 0) + 1

        avg_E: Dict[str, float] = {}
        for fk in sum_E.keys():
            n = cnt_E.get(fk, 0)
            if n > 0:
                avg_E[fk] = sum_E[fk] / float(n)

        ratios = _ratio_vs_ha_from_E(avg_E) if avg_E else {}

        # Aggregate score is not meaningful (score depends on penalties across candidate set),
        # but users want a stable ratio. We still show E_mean/E_min computed from avg_E.
        E_vals = list(avg_E.values())
        E_mean = (sum(E_vals) / float(len(E_vals))) if E_vals else None
        E_min = min(E_vals) if E_vals else None

        agg = {
            "label": "AGGREGATE",
            "exptimes_used": sorted(set(exp_list)),
            "E_per_filter_avg": avg_E,
            "E_mean": E_mean,
            "E_min": E_min,
            "min_linear_headroom_p99": min_head,
            "ratio_vs_ha": ratios,
            "note": "Aggregate is averaged across all eligible exposures (stabilizes ratios when you only have 1 frame per filter per exposure).",
        }

    out: Dict[str, Any] = {}
    if best is not None:
        out["best"] = best
    if second is not None:
        out["second_best"] = second
        # Add delta if best exists
        if (
            best is not None
            and best.get("score") is not None
            and second.get("score") is not None
        ):
            try:
                out["second_best"]["delta_vs_best"] = float(best["score"]) - float(
                    second["score"]
                )
            except Exception:
                pass
    if agg is not None:
        out["aggregate"] = agg
    return out


# -----------------------------
# Report builder
# -----------------------------


def _build_report_text(payload: Dict[str, Any]) -> str:
    """
    Human-friendly multiline report for logs/CLI.
    """
    target = str(payload.get("target_name") or "").strip()
    rec = payload.get("recommended") or {}
    candidates = payload.get("candidates") or []
    excluded = payload.get("excluded") or {}
    comparisons = payload.get("comparisons") or {}

    rx = rec.get("exptime_s")
    ratios = rec.get("ratio_vs_ha") or {}
    one_liner = rec.get("one_liner") or ""

    lines: List[str] = []
    lines.append(
        f"TRAINING RECOMMENDATION — {target}" if target else "TRAINING RECOMMENDATION"
    )

    if rx is not None:
        try:
            lines.append(f"RECOMMENDED: {int(round(float(rx)))}s")
        except Exception:
            lines.append(f"RECOMMENDED: {rx}")

    ratio_line = _stable_ratio_line(dict(ratios))
    if ratio_line:
        lines.append("RATIO vs HA: " + ratio_line)

    if one_liner:
        lines.append(f"SUMMARY: {one_liner}")

    # NEW: comparisons section (best / 2nd / aggregate)
    if comparisons:
        lines.append("")
        lines.append("RATIO COMPARISON (Best / 2nd / Aggregate)")

        best = comparisons.get("best") or {}
        second = comparisons.get("second_best") or {}
        agg = comparisons.get("aggregate") or {}

        def fmt_row(
            label: str,
            exp: Any,
            ratio: Dict[str, Any],
            E_mean: Any,
            E_min: Any,
            head: Any,
            extra: str = "",
        ) -> str:
            exp_s = "—"
            if exp is not None:
                try:
                    exp_s = f"{int(round(float(exp)))}s"
                except Exception:
                    exp_s = str(exp)
            rline = _stable_ratio_line(dict(ratio or {})) or "—"
            return (
                f"  {label:<9} {exp_s:>5} | "
                f"{rline} | "
                f"E_mean {_fmt_float(E_mean, nd=3):>6} "
                f"E_min {_fmt_float(E_min, nd=3):>6} "
                f"head_min {_fmt_float(head, nd=3):>6}"
                + (f" | {extra}" if extra else "")
            )

        if best:
            lines.append(
                fmt_row(
                    "BEST",
                    best.get("exptime_s"),
                    best.get("ratio_vs_ha"),
                    best.get("E_mean"),
                    best.get("E_min"),
                    best.get("min_linear_headroom_p99"),
                )
            )

        if second:
            extra = ""
            dvb = second.get("delta_vs_best")
            if dvb is not None:
                try:
                    extra = f"Δscore {float(dvb):.3f}"
                except Exception:
                    extra = ""
            lines.append(
                fmt_row(
                    "2ND",
                    second.get("exptime_s"),
                    second.get("ratio_vs_ha"),
                    second.get("E_mean"),
                    second.get("E_min"),
                    second.get("min_linear_headroom_p99"),
                    extra=extra,
                )
            )

        if agg:
            extra = ""
            exps_used = agg.get("exptimes_used") or []
            if exps_used:
                extra = "exps " + ",".join([str(x) for x in exps_used])
            lines.append(
                fmt_row(
                    "AGGREGATE",
                    "—",
                    agg.get("ratio_vs_ha"),
                    agg.get("E_mean"),
                    agg.get("E_min"),
                    agg.get("min_linear_headroom_p99"),
                    extra=extra,
                )
            )
            note = str(agg.get("note") or "").strip()
            if note:
                lines.append(f"    ↳ {note}")

    lines.append("")
    lines.append("CANDIDATES")

    # Column header
    lines.append("  exp   pick     score   E_mean  E_min   headroom_min  filters  note")
    for c in candidates:
        exp = c.get("exptime_s")
        exp_s = f"{int(exp):>3d}s" if isinstance(exp, int) else f"{str(exp):>4}"
        chosen = bool(c.get("chosen"))
        reason = str(c.get("reason") or "").strip()

        score = c.get("score")
        E_mean = c.get("E_mean")
        E_min = c.get("E_min")
        head = c.get("min_linear_headroom_p99")

        if chosen:
            pick = "✅ chosen "
        else:
            if c.get("excluded_reasons") or score is None:
                pick = "❌ excl. "
            else:
                pick = "❌ not   "

        filt = _fmt_filters(c.get("filters"))

        row = (
            f"  {exp_s}  {pick}"
            f" {_fmt_score(score):>6} "
            f" {_fmt_float(E_mean):>6} "
            f" {_fmt_float(E_min):>6} "
            f" {_fmt_float(head):>12}  "
            f"{filt:>7}  "
            f"{reason}"
        )
        lines.append(row)

        exr = c.get("excluded_reasons")
        if exr:
            try:
                exr_list = list(exr)
            except Exception:
                exr_list = [str(exr)]
            for r in exr_list:
                lines.append(f"           ↳ {r}")

    if excluded:
        lines.append("")
        lines.append("EXCLUDED (raw)")
        for k in _sort_exposure_keys(list(excluded.keys())):
            lines.append(f"  {k}s: {', '.join([str(x) for x in excluded[k]])}")

    return "\n".join(lines)


# -----------------------------
# Main solve wrapper
# -----------------------------


def solve_session_and_optionally_close(
    *,
    db_path: str,
    session_id: int,
    close_session: bool,
) -> Tuple[bool, Dict[str, Any]]:
    cfg = default_config()

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON;")

        result = solve_training_session(
            con,
            int(session_id),
            required_filters_default=list(
                getattr(cfg, "training_required_filters_default", ["SII", "HA", "OIII"])
            ),
            min_linear_headroom_default=float(
                getattr(cfg, "training_min_linear_headroom_p99", 0.10)
            ),
            w_mean_default=float(getattr(cfg, "training_score_w_mean", 0.55)),
            w_min_default=float(getattr(cfg, "training_score_w_min", 0.45)),
            short_penalty_default=float(getattr(cfg, "training_short_penalty", 0.10)),
            long_penalty_default=float(getattr(cfg, "training_long_penalty", 0.08)),
            include_excluded_default=bool(
                getattr(cfg, "training_include_excluded", True)
            ),
            close_call_delta_default=float(
                getattr(cfg, "training_close_call_delta", 0.05)
            ),
        )

        if not result.get("ok"):
            return False, {
                "error": result.get("message") or "solve_failed",
                "details": result,
            }

        if close_session:
            with con:
                close_training_session(con, int(session_id))

        session = result.get("session") or {}
        best = result.get("best") or {}
        exposure_report = result.get("exposure_report") or {}
        excluded = result.get("excluded") or {}

        recommended_exptime_s = best.get("recommended_exptime_s")
        ratio_vs_ha = best.get("ratio_vs_ha") or {}
        time_fractions = best.get("time_fractions") or {}

        one_liner = ""
        if recommended_exptime_s is not None:
            rx = float(recommended_exptime_s)
            ratio_str = " ".join(
                [f"{k}:{float(v):.2f}x" for k, v in ratio_vs_ha.items()]
            )
            one_liner = f"{rx:.0f}s | {ratio_str}".strip()

        # Build candidates list from exposure_report (already assembled by engine)
        candidates: List[Dict[str, Any]] = []
        for exp_key in _sort_exposure_keys(list(exposure_report.keys())):
            rep = exposure_report.get(exp_key) or {}
            candidates.append(
                {
                    "exptime_s": (
                        int(float(exp_key))
                        if str(exp_key).replace(".", "", 1).isdigit()
                        else exp_key
                    ),
                    "filters": rep.get("filters"),
                    "min_linear_headroom_p99": rep.get("min_linear_headroom_p99"),
                    "E_mean": rep.get("E_mean"),
                    "E_min": rep.get("E_min"),
                    "score": rep.get("score"),
                    "frames_used": rep.get("frames_used"),
                    "chosen": rep.get("chosen"),
                    "reason": rep.get("reason"),
                    "excluded_reasons": rep.get("excluded_reasons"),
                }
            )

        # NEW: comparisons (best / 2nd / aggregate ratios)
        comparisons = _build_comparisons_from_exposure_report(exposure_report)

        summary: Dict[str, Any] = {
            "session_id": int(session_id),
            "target_name": session.get("target_name", ""),
            "recommended": {
                "exptime_s": recommended_exptime_s,
                "ratio_vs_ha": ratio_vs_ha,
                "time_fractions": time_fractions,
                "score": best.get("score"),
                "E_mean": best.get("E_mean"),
                "E_min": best.get("E_min"),
                "frames_used": best.get("frames_used"),
                "one_liner": one_liner,
            },
            "comparisons": comparisons,
            "candidates": candidates,
        }

        if excluded:
            summary["excluded"] = excluded

        summary = _round_recursive(summary, nd=3)
        summary["report_text"] = _build_report_text(summary)

        return True, summary

    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Solve a training session and optionally close it."
    )
    ap.add_argument("--db", required=True, help="Path to sqlite DB")
    ap.add_argument("--session-id", type=int, required=True, help="training_session_id")
    ap.add_argument("--close", action="store_true", help="Close session after solving")
    args = ap.parse_args()

    ok, payload = solve_session_and_optionally_close(
        db_path=str(args.db),
        session_id=int(args.session_id),
        close_session=bool(args.close),
    )

    if ok:
        print(
            payload.get("report_text")
            or (payload.get("recommended") or {}).get("one_liner")
            or "ok"
        )
        return 0

    print(f"ERROR: {payload.get('error')}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
